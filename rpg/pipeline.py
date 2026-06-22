"""The per-turn world pipeline, as a LangGraph state machine.

parse -> retrieve context -> validate -> resolve -> guardrail -> update -> narrate,
with a refusal branch when an action is illegal and a bounded retry when the model
proposes an outcome the graph can't accept. The model proposes; the graph disposes.
Consistency is enforced here in code: validate rejects illegal moves and applies
declarative gates, guardrail rejects fabricated outcomes, and narrate is policed so
it can only name entities actually present. The active GameState rides through the
graph under the ``gs`` key so nodes mutate the real session, not a global.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import config
from .players import refresh_quests
from .schemas import Intent, Resolution
from .world import GATES, G, world_context

BANNED = ("quest", "quests", "xp", "respawn", "cooldown", "hitpoints", "gameplay", "sidequest")
MAX_RETRIES = 2

PARSE_SYS = (
    "Convert the player's text into a structured action. action is one of move, take, talk, look, use, say. "
    "Treat buy, purchase, or grab as take. "
    "Use 'say' for talking to companions or general speech (questions, reactions, chit-chat): put the words "
    "in 'message' and any addressed name in 'target'. Use 'talk' when addressing an NPC. For move/take, "
    "'target' is a single lowercase noun. Otherwise 'message' is empty."
)


def parse_intent(state):
    gs = state["gs"]
    out = state.get("intent")  # reuse a pre-parsed intent if the caller already classified it
    if not out:
        out = config.work_struct(Intent, [("system", PARSE_SYS), ("human", state["player_input"])])
    if out.get("action") == "move" and (out.get("target") or "").lower() in ("back", "return", "previous", "backward"):
        out["target"] = gs.prev_location or out.get("target")  # 'go back' -> last room
    return {"intent": out}


def retrieve_context(state):
    gs = state["gs"]
    ctx = world_context(gs, gs.location)
    ctx["actor"] = state.get("actor", "the party")
    return {"context": ctx}


def validate_action(state):
    intent, ctx = state["intent"], state["context"]
    action = (intent.get("action") or "look").lower()
    target = (intent.get("target") or "").lower().strip()
    valid, reason = True, ""
    if action == "move" and target not in ctx["exits"]:
        valid, reason = False, f"There is no path to '{target}' from here."
    elif action == "take" and target not in ctx["items_here"]:
        valid, reason = False, f"There is no {target} here to take."
    elif action == "talk" and target not in ctx["npcs_here"]:
        valid, reason = False, f"There is no {target} here to talk to."
    elif action not in ("move", "take", "talk", "look", "use"):
        valid, reason = False, "You can't do that."
    if valid:  # structurally ok -> apply any declarative gate
        gate = GATES.get(f"{action}:{target}")
        if gate and gate["need"] not in ctx["inventory"]:
            valid, reason = False, gate["reason"]
    return {"valid": valid, "reason": reason}


def resolve_outcome(state):
    sysm = ("You are the game engine. Given a legal action and context, produce the concrete outcome. "
            "Only move to a listed exit; only grant an item listed as present; invent nothing.")
    usr = json.dumps({"action": state["intent"], "context": state["context"]})
    d = config.work_struct(Resolution, [("system", sysm), ("human", usr)])
    return {"delta": d, "retries": state.get("retries", 0)}


def guardrail_check(state):
    d, ctx = state["delta"], state["context"]
    problems = []
    if d["move_to"] and d["move_to"] not in ctx["exits"]:
        problems.append(f"illegal move -> {d['move_to']}")
    if d["grant_item"] and d["grant_item"] not in ctx["items_here"]:
        problems.append(f"absent item -> {d['grant_item']}")
    retries = state.get("retries", 0)
    if problems and retries < MAX_RETRIES:
        return {"guardrail_ok": False, "retries": retries + 1}
    if problems:  # exhausted retries -> a safe no-op the narrator can't twist
        return {"guardrail_ok": True, "delta": {"move_to": None, "grant_item": None, "note": "Nothing happens."}}
    return {"guardrail_ok": True}


def update_state(state):
    gs = state["gs"]
    d = state["delta"]
    if d["move_to"]:
        gs.prev_location = gs.location
        gs.location = d["move_to"]
        gs.visited.add(d["move_to"])
    if d["grant_item"] and d["grant_item"] not in gs.inventory:
        gs.inventory.append(d["grant_item"])
    refresh_quests(gs)
    return {}


def narr_view(c):  # the narrator sees the room and its contents, not exits or quest bookkeeping
    return {k: c.get(k) for k in ("room", "room_desc", "items_here", "npcs_here", "inventory") if k in c}


def world_entity_names():
    return [n for n, a in G.nodes(data=True) if a.get("type") in ("room", "item", "npc", "enemy")]


def narration_ok(text, allowed):
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & set(BANNED):  # whole-word match avoids request/conquest/points false hits
        return False
    return not any(n not in allowed and n in words for n in world_entity_names())


def narrate(state):
    gs = state["gs"]
    ctx = world_context(gs, gs.location)
    prev = state.get("context", {})  # full pre-move context the pipeline already captured
    actor = state.get("actor", "the party")
    allowed = set([ctx["room"], prev.get("room", ctx["room"])]
                  + ctx["items_here"] + ctx["npcs_here"] + ctx["inventory"]
                  + prev.get("items_here", []) + prev.get("npcs_here", [])
                  + [p["name"].lower() for p in gs.party])
    sysm = ("You are a concise fantasy narrator. 1-2 sentences, third person. Describe the party's CURRENT "
            f"room (in 'now') and what {actor} just did (in 'happened'); name the character. You may also "
            "reference 'before' (the room they just left and what was in it). Never invent any other place, "
            "exit, item, or character, never mention quests, scores, or mechanics, and never name anything "
            "not in 'now', 'before', or 'happened'.")
    usr = json.dumps({"actor": actor, "happened": state["delta"],
                      "now": narr_view(ctx), "before": narr_view(prev)})
    text = None
    for _ in range(3):  # regenerate if it names an out-of-scope entity or leaks mechanics
        cand = config.work_text([("system", sysm), ("human", usr)], max_tokens=90, temperature=0.8)
        if narration_ok(cand, allowed):
            text = cand
            break
    if text is None:  # every attempt leaked -> a safe template that can't
        text = f"{actor} stands in the {ctx['room']}."
    return {"narration": text, "post_context": ctx}


def narrate_refusal(state):
    sysm = ("You are a fantasy narrator. One sentence, third person: tell the actor they can't do that, using "
            "the given reason. Stay in character; invent nothing.")
    usr = json.dumps({"actor": state.get("actor", "the party"), "reason": state.get("reason", "")})
    text = config.work_text([("system", sysm), ("human", usr)], max_tokens=50)
    return {"narration": text, "post_context": world_context(state["gs"], state["gs"].location)}


class TurnState(TypedDict, total=False):
    gs: Any
    player_input: str
    actor: str
    intent: dict[str, Any]
    context: dict[str, Any]
    valid: bool
    reason: str
    delta: dict[str, Any]
    guardrail_ok: bool
    retries: int
    narration: str
    post_context: dict[str, Any]


def _route_validity(state):
    return "resolve_outcome" if state["valid"] else "narrate_refusal"


def _route_guardrail(state):
    return "update_state" if state["guardrail_ok"] else "resolve_outcome"


def _build_app():
    g = StateGraph(TurnState)
    for n, fn in [("parse_intent", parse_intent), ("retrieve_context", retrieve_context),
                  ("validate_action", validate_action), ("resolve_outcome", resolve_outcome),
                  ("guardrail_check", guardrail_check), ("narrate", narrate),
                  ("narrate_refusal", narrate_refusal), ("update_state", update_state)]:
        g.add_node(n, fn)
    g.set_entry_point("parse_intent")
    g.add_edge("parse_intent", "retrieve_context")
    g.add_edge("retrieve_context", "validate_action")
    g.add_conditional_edges("validate_action", _route_validity,
                            {"resolve_outcome": "resolve_outcome", "narrate_refusal": "narrate_refusal"})
    g.add_edge("resolve_outcome", "guardrail_check")
    g.add_conditional_edges("guardrail_check", _route_guardrail,
                            {"update_state": "update_state", "resolve_outcome": "resolve_outcome"})
    g.add_edge("update_state", "narrate")
    g.add_edge("narrate", END)
    g.add_edge("narrate_refusal", END)
    return g.compile()


app = _build_app()


def run_turn(gs, player_input: str, actor: str, intent: dict | None = None) -> dict:
    """Run one world action through the pipeline against ``gs`` and return the merged result."""
    payload = {"gs": gs, "player_input": player_input, "actor": actor, "retries": 0}
    if intent:
        payload["intent"] = intent
    return app.invoke(payload)
