"""LLM-as-judge faithfulness grading.

A higher-tier model (``config.judge_struct``) grades generated narration and
dialogue against the ground-truth world facts: did it invent an entity, leak a
mechanic, break character, or contradict state? This catches the semantic slips
the deterministic guardrail cannot see. Grading is advisory; it never gates play,
it only scores output for evaluation. Tests stub ``judge_struct``, so the wiring
runs with no API key.
"""
from __future__ import annotations

import json

from . import config
from .pipeline import world_entity_names
from .schemas import Verdict
from .world import G

JUDGE_SYS = (
    "You are a strict faithfulness judge for the game master of a text RPG. Given one line of "
    "generated text and the ground-truth facts of the world, score how faithful the line is from "
    "0 (egregious) to 10 (clean), and name every violation you find in 'violations' using exactly "
    "these labels: 'invented_entity' (a place, item, person, or creature not in facts.entities), "
    "'leaked_mechanic' (states a quest, score, hit points, mana, dice, turn, or other game term as a "
    "mechanic), 'out_of_character' (a modern or out-of-world voice for a fantasy narrator or "
    "speaker), 'contradiction' (asserts something the facts mark false). Judge only faithfulness to "
    "the facts, never prose quality. Keep 'reason' to one short clause."
)


def world_facts():
    """The ground-truth vocabulary a faithful line may draw on: every real entity, grouped by type."""
    facts = {"rooms": [], "items": [], "npcs": [], "enemies": []}
    key = {"room": "rooms", "item": "items", "npc": "npcs", "enemy": "enemies"}
    for name in world_entity_names():
        facts[key[G.nodes[name]["type"]]].append(name)
    return facts


def grade(text, facts=None, kind="narration"):
    """Grade one generated line against the world facts; returns a Verdict dict (score, violations, reason)."""
    facts = facts if facts is not None else world_facts()
    usr = json.dumps({"kind": kind, "text": text, "entities": facts})
    return config.judge_struct(Verdict, [("system", JUDGE_SYS), ("human", usr)])


def summarize(verdicts):
    """Aggregate graded verdicts into a compact report: count, mean score, clean rate, violation tally."""
    n = len(verdicts)
    if not n:
        return {"n": 0, "mean_score": None, "clean_rate": None, "violations": {}}
    scores = [int(v.get("score", 0)) for v in verdicts]
    tally: dict[str, int] = {}
    for v in verdicts:
        for name in v.get("violations", []):
            tally[name] = tally.get(name, 0) + 1
    clean = sum(1 for v in verdicts if not v.get("violations"))
    return {"n": n, "mean_score": round(sum(scores) / n, 2),
            "clean_rate": round(clean / n, 3), "violations": tally}
