"""Agent reasoning: the compass, discovery-gated goals, per-turn decisions, and the
no-human navigation vote.

The graph computes the correct next objective and route; agents read it but only
learn where things are once the party discovers them. With no human present, each
agent argues for a destination and a deterministic, assertiveness-weighted vote
resolves it, with the compass as a baseline voice so a confident-but-wrong
consensus needs real support to override the proven route.
"""
from __future__ import annotations

import json

import networkx as nx

from . import config
from .events import Argument
from .players import clamp
from .schemas import AgentTurn, Proposal
from .world import G, exits, npcs_at, world_context

QUEST_HINTS = {
    "retrieve_amulet": "the amulet is in the ruins behind a locked shrine; its key lies in the cave past the forest, guarded by a wolf",
    "slay_guardian": "defeat the guardian in the crypt; it is dark, so bring the torch from the market",
}
VAGUE_HINTS = {
    "retrieve_amulet": "retrieve the amulet, though where it lies and what reaching it demands are still unknown to you",
    "slay_guardian": "slay the guardian, though where it lurks is still unknown to you",
}


def roster(gs, player):
    """How the other party members read to one agent, tagged human or ally."""
    out = []
    for p in gs.party:
        if p is player:
            continue
        tag = "the human player" if not p["is_agent"] else "an ally"
        out.append(f"{p['name']} the {p['class_name']} ({tag})")
    return ", ".join(out) or "no one else"


def open_goals(gs):
    """The full quest hints (the NPC's knowledge)."""
    return [QUEST_HINTS.get(q, q) for q, v in gs.quests.items() if v != "done"]


def known_goals(gs):
    """Goals as the party understands them: full hint once they hold intel, else vague."""
    src = QUEST_HINTS if gs.intel else VAGUE_HINTS
    return [src.get(q, q) for q, v in gs.quests.items() if v != "done"]


def node_room(name):
    """The room a non-room node (item, npc, enemy) lives in."""
    for _, r, e in G.out_edges(name, data=True):
        if e["type"] in ("in", "guards", "at"):
            return r
    return None


def next_objective(gs):
    """The single most useful thing the party still needs, derived from quest state."""
    inv = gs.inventory
    if "amulet" not in inv:
        return ("the key", node_room("key")) if "key" not in inv else ("the amulet", node_room("amulet"))
    if "guardian" not in gs.defeated:
        return ("the torch", node_room("torch")) if "torch" not in inv else ("the guardian", node_room("guardian"))
    return (None, None)


def heading(gs):
    """Objective and the next room toward it, but only once the party has discovered where it is."""
    obj, dest = next_objective(gs)
    if not dest:
        return {"target": None, "go_to": None, "known": True}
    known = gs.intel or dest in gs.visited
    if dest == gs.location:
        return {"target": obj, "go_to": None, "known": True}
    if not known:
        return {"target": obj, "go_to": None, "known": False}
    try:
        path = nx.shortest_path(G, gs.location, dest)
        return {"target": obj, "go_to": (path[1] if len(path) > 1 else None), "known": True}
    except Exception:
        return {"target": obj, "go_to": None, "known": True}


def agent_decide(gs, player, can_move=True):
    """An agent picks one action and a line, steered by what's discovered and whether it leads the march."""
    h = heading(gs)
    here = [p["name"] for p in gs.alive() if p is not player] + [n.capitalize() for n in npcs_at(gs.location)]
    present = ", ".join(here) or "no one else"
    base = (f"You are {player['name']}, {player['class_desc']}, who is {player['personality']}, adventuring with "
            f"{roster(gs, player)}. Present with you right now: {present}. React to what just happened, then issue "
            "exactly ONE action (no lists, semicolons, or 'then'). Address only someone present by name; if you are "
            "alone, do not invent a team or address absent allies or an NPC who is not in this room. You do not know "
            "the way ahead in advance: learn what a place or prize demands by asking those who know or by trying and "
            "being turned back, then go fetch what is needed. A potion is your only heal in a fight and the road has "
            "enemies, so if one is here and the party carries none, take it. Never re-ask something already "
            "answered. Say one short in-character line. ")
    if can_move:
        rule = ("Use 'objective' as your compass: if 'go_to' is set, travel there and nowhere else; if 'known' is "
                "false you do not yet know where to look, so ask a knowledgeable NPC such as the elder if one is "
                "here, and only wander an unvisited path if there is no one to ask.")
    else:
        rule = ("The party travels together and you are not leading the march this turn, so do NOT travel. Do "
                "something useful where you stand: take a potion or other useful item that is here, talk to a "
                "companion or an NPC, or hold position.")
    usr = json.dumps({"now": world_context(gs, gs.location), "goals": known_goals(gs),
                      "objective": h, "carrying_potion": "potion" in gs.inventory, "recent": gs.recent_memory()})
    out = config.work_struct(AgentTurn, [("system", base + rule), ("human", usr)], temperature=0.85)
    return (out.get("say", "") or "").strip(), (out.get("action", "look") or "look").strip()


def resolve_proposals(proposals, compass_go_to=None, compass_weight=3):
    """Pick the party destination by assertiveness-weighted vote; the compass gets a baseline voice."""
    if not proposals:
        return "stay"
    tally = {}
    if compass_go_to:
        tally[compass_go_to] = compass_weight  # the proven-correct route is not just a tie-break
    for _, dest, _, w in proposals:
        tally[dest] = tally.get(dest, 0) + w
    top = max(tally.values())
    leaders = [d for d, t in tally.items() if t == top]
    if len(leaders) == 1:
        return leaders[0]
    if compass_go_to in leaders:
        return compass_go_to
    return sorted(leaders)[0]


def negotiate_move(gs):
    """No-human navigation: each agent argues for a destination, the engine tallies. Yields Argument, returns (dest, proposals)."""
    opts = exits(gs.location) + ["stay"]
    h = heading(gs)
    proposals = []
    for p in [a for a in gs.alive() if a["is_agent"]]:
        sysm = (f"You are {p['name']}, a {p['caution']} {p['combat_focus']} adventurer. The party decides together "
                "where to go next. The compass's 'go_to' is the known-correct next step toward the goal, so propose "
                "that UNLESS your nature gives a concrete reason to deviate: the cautious may want to fall back for "
                "light or healing before danger, the bold may want to press on. Do not propose a room because you "
                "guess it connects somewhere, trust the compass for geography. Propose ONE option, or 'stay', with "
                "one short reason.")
        usr = json.dumps({"options": opts, "compass": h, "goals": known_goals(gs),
                          "here": world_context(gs, gs.location), "recent": gs.recent_memory()})
        prop = config.work_struct(Proposal, [("system", sysm), ("human", usr)])
        dest = (prop.get("destination") or "stay").lower().strip()
        if dest not in opts:
            dest = "stay"
        w = clamp(p.get("assertiveness", 3), 1, 5, 3)
        reason = (prop.get("reason") or "").strip()
        proposals.append((p["name"], dest, reason, w))
        gs.remember(f"{p['name']} argues to head to {dest}")
        yield Argument(p["name"], dest, reason)
    return resolve_proposals(proposals, h.get("go_to")), proposals
