"""The evaluation harness.

Two layers, matching the two defenses in the engine:

- A deterministic robustness suite (no API key). It fabricates every illegal
  outcome the model could propose, a teleport to a non-adjacent room and a grant of
  an absent item, in every room, and asserts the pipeline guardrail both detects and
  neutralizes each one; it also feeds crafted leaks to the narration policer. These
  are exhaustive, fast, and the hard numbers the README cites.

- An LLM-judge faithfulness pass (key required). It plays a game and grades the
  narration and dialogue it produces with ``judge.grade``, catching semantic slips
  the code cannot. Skipped with a clear note when no key is set.

Run with ``rpg-eval`` or ``python -m rpg.evals``.
"""
from __future__ import annotations

import json

from . import config, judge, players
from .events import Dialogue, Narration
from .pipeline import BANNED, MAX_RETRIES, guardrail_check, narration_ok, run_turn, world_entity_names
from .world import WORLD, G, world_context


def _items():
    return [n for n, a in G.nodes(data=True) if a.get("type") == "item"]


def _fabrications(ctx):
    """Every illegal outcome a cheating model could return for a legal action in this room's context."""
    rooms = list(WORLD["rooms"])
    for r in rooms:
        if r != ctx["room"] and r not in ctx["exits"]:
            yield {"move_to": r, "grant_item": None, "note": "teleport"}
    for it in _items():
        if it not in ctx["items_here"]:
            yield {"move_to": None, "grant_item": it, "note": "phantom"}


def _legal_outcomes(ctx):
    """Outcomes that are actually legal here, used to confirm the guardrail does not over-block."""
    for r in ctx["exits"]:
        yield {"move_to": r, "grant_item": None, "note": "ok"}
    for it in ctx["items_here"]:
        yield {"move_to": None, "grant_item": it, "note": "ok"}


def guardrail_eval():
    """Fault-inject the guardrail in every room: it must detect every fabrication, contain it, and pass legals."""
    gs = players.new_game([players.make_player("Probe", "probe", stats={"name": "p", "max_hp": 20, "attack": 8})])
    detected = contained = injected = 0
    false_positives = legal = 0
    for room in WORLD["rooms"]:
        ctx = world_context(gs, room)
        for d in _fabrications(ctx):
            injected += 1
            if guardrail_check({"delta": d, "context": ctx, "retries": 0})["guardrail_ok"] is False:
                detected += 1  # caught on the first pass and sent back for a retry
            final = guardrail_check({"delta": d, "context": ctx, "retries": MAX_RETRIES}).get("delta", d)
            if final.get("move_to") is None and final.get("grant_item") is None:
                contained += 1  # retries exhausted -> reduced to a safe no-op
        for d in _legal_outcomes(ctx):
            legal += 1
            if guardrail_check({"delta": d, "context": ctx, "retries": 0})["guardrail_ok"] is False:
                false_positives += 1
    return {"injected": injected, "detected": detected, "contained": contained,
            "legal": legal, "false_positives": false_positives}


def narration_policer_eval():
    """Feed the narration policer crafted leaks and clean lines; it must reject the leaks and pass the clean."""
    gs = players.new_game([players.make_player("Probe", "probe", stats={"name": "p", "max_hp": 20, "attack": 8})])
    ctx = world_context(gs, "village")
    allowed = set([ctx["room"]] + ctx["items_here"] + ctx["npcs_here"] + ctx["inventory"]
                  + [p["name"].lower() for p in gs.party])
    offscope = [n for n in world_entity_names() if n not in allowed]
    leaks = [f"A {n} looms suddenly out of the dark." for n in offscope]                # invented entity
    leaks += [f"You gain a {w} as the scene resolves." for w in BANNED]                  # leaked mechanic
    clean = [f"The {ctx['room']} lies quiet around {p['name'].lower()}." for p in gs.party]
    clean += ["A cold wind moves through the empty square.", "The elder watches in silence."]
    caught = sum(1 for t in leaks if not narration_ok(t, allowed))
    passed = sum(1 for t in clean if narration_ok(t, allowed))
    return {"leaks": len(leaks), "leaks_caught": caught, "clean": len(clean), "clean_passed": passed}


def endtoend_guardrail_eval():
    """Drive the whole pipeline with a model that always cheats; the world state must never accept it."""
    rooms = list(WORLD["rooms"])

    def cheat(usr, phantom):
        d = json.loads(usr)
        ctx = d["context"]
        if phantom:
            it = next((i for i in _items() if i not in ctx["items_here"]), None)
            return {"move_to": None, "grant_item": it, "note": "cheat"}
        bad = next((r for r in rooms if r != ctx["room"] and r not in ctx["exits"]), None)
        return {"move_to": bad, "grant_item": None, "note": "cheat"}

    state = {"phantom": False}

    def stub_struct(schema, messages, temperature=None):
        name = schema.__name__
        usr = messages[-1][1]
        if name == "Intent":
            return {"action": "look", "target": "", "message": ""}
        if name == "Resolution":
            return cheat(usr, state["phantom"])
        return {}

    orig_struct, orig_text = config.work_struct, config.work_text
    config.work_struct = stub_struct
    config.work_text = lambda messages, max_tokens=160, temperature=None, label="work_text": "A still moment passes in the gloom."
    blocked = total = 0
    try:
        for phantom in (False, True):
            state["phantom"] = phantom
            for room in rooms:
                gs = players.new_game([players.make_player("P", "p", stats={"name": "p", "max_hp": 20, "attack": 8})])
                gs.location = room
                gs.visited = {room}
                before = (gs.location, tuple(gs.inventory))
                res = run_turn(gs, "look", "P")
                total += 1
                unchanged = (gs.location, tuple(gs.inventory)) == before
                if unchanged and narration_ok(res.get("narration", ""), {room}):
                    blocked += 1
    finally:
        config.work_struct, config.work_text = orig_struct, orig_text
    return {"turns": total, "blocked": blocked}


def robustness_report():
    """Run all deterministic checks and return a single results dict (no API key needed)."""
    return {"guardrail": guardrail_eval(),
            "narration_policer": narration_policer_eval(),
            "endtoend": endtoend_guardrail_eval()}


def _collect_game_text(max_rounds, cap):
    """Play one all-agent game and return up to ``cap`` generated narration/dialogue lines, with kinds."""
    from .engine import play
    gs = players.new_game([
        players.make_player("Borin", "a vanguard", "bold", is_agent=True,
                            stats={"name": "vanguard", "max_hp": 36, "attack": 12, "assertiveness": 4}),
        players.make_player("Sable", "a battle-mage", "sharp", is_agent=True,
                            stats={"name": "mage", "max_hp": 30, "attack": 12, "max_mana": 12,
                                   "combat_focus": "offense", "assertiveness": 3}),
    ])
    lines = []
    gen = play(gs, max_rounds=max_rounds)
    to_send = None
    while len(lines) < cap:
        try:
            ev = gen.send(to_send)
        except StopIteration:
            break
        to_send = None
        if isinstance(ev, Narration):
            lines.append(("narration", ev.text))
        elif isinstance(ev, Dialogue):
            lines.append(("dialogue", ev.text))
    return lines


def judge_eval(max_rounds=12, cap=24):
    """Play a live game and grade its narration and dialogue for faithfulness. Needs an API key."""
    if not config.has_key():
        return {"skipped": "no OPENAI_API_KEY; the faithfulness judge needs a live model"}
    facts = judge.world_facts()
    verdicts = [judge.grade(text, facts, kind=kind) for kind, text in _collect_game_text(max_rounds, cap)]
    return judge.summarize(verdicts)


def _fmt(report):
    g, n, e = report["guardrail"], report["narration_policer"], report["endtoend"]
    return "\n".join([
        "== robustness (deterministic, no key) ==",
        f"guardrail: detected {g['detected']}/{g['injected']} fabrications, "
        f"contained {g['contained']}/{g['injected']}, "
        f"false positives {g['false_positives']}/{g['legal']} legal outcomes",
        f"narration policer: caught {n['leaks_caught']}/{n['leaks']} leaks, "
        f"passed {n['clean_passed']}/{n['clean']} clean lines",
        f"end-to-end cheating model: blocked {e['blocked']}/{e['turns']} turns",
    ])


def main():
    report = robustness_report()
    print(_fmt(report))
    print("\n== faithfulness judge (live model) ==")
    jr = judge_eval()
    if "skipped" in jr:
        print(f"skipped: {jr['skipped']}")
    else:
        print(f"graded {jr['n']} lines | mean score {jr['mean_score']}/10 | "
              f"clean rate {jr['clean_rate']} | violations {jr['violations'] or 'none'}")


if __name__ == "__main__":
    main()
