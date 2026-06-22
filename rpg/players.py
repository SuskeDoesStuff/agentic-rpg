"""Players, derived from free text.

A class description and personality go to the model in one call, which returns
stats (hp, attack, mana) and a disposition (combat focus from the class, caution
and council assertiveness from the personality). Code clamps everything to safe
ranges, so nothing the model returns can break balance or invariants. Pass
explicit ``stats=`` to skip the call, which is how the demo and tests stay
deterministic.
"""
from __future__ import annotations

from . import config
from .schemas import ClassStats
from .state import GameState

HP_MIN, HP_MAX, ATK_MIN, ATK_MAX, MANA_MAX = 12, 36, 4, 12, 20


def clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return default


def gen_class_stats(desc, personality=""):
    """Derive stats and a disposition from a free-text class and personality."""
    sysm = (
        "Given a fantasy character's class description and personality, assign balanced RPG stats and a "
        f"disposition. max_hp {HP_MIN}-{HP_MAX} (tanky higher); attack {ATK_MIN}-{ATK_MAX} (glass-cannon "
        f"higher); max_mana 0-{MANA_MAX} (0 for pure martials, high for dedicated casters). combat_focus is "
        "offense, support, or balanced, drawn from the class. caution is cautious, steady, or reckless, "
        "blending the character's fragility with their personality. assertiveness is 1 (defers) to 5 "
        "(dominates) in group decisions, drawn from personality. Give a short class name too."
    )
    human = f"class: {desc}\npersonality: {personality or 'unspecified'}"
    try:
        return config.work_struct(ClassStats, [("system", sysm), ("human", human)])
    except Exception:
        return {"name": desc[:20], "max_hp": 22, "attack": 7, "max_mana": 0,
                "combat_focus": "balanced", "caution": "steady", "assertiveness": 3}


def make_player(name, class_desc, personality="", is_agent=False, stats=None):
    """Build a player; free text becomes clamped stats plus a disposition unless stats are given."""
    s = stats or gen_class_stats(class_desc, personality)
    hp = clamp(s.get("max_hp"), HP_MIN, HP_MAX, 22)
    mana = clamp(s.get("max_mana", 0), 0, MANA_MAX, 0)
    return {"name": name, "class_desc": class_desc, "class_name": s.get("name", class_desc),
            "personality": personality, "is_agent": is_agent,
            "attack": clamp(s.get("attack"), ATK_MIN, ATK_MAX, 7), "hp": hp, "max_hp": hp,
            "mana": mana, "max_mana": mana,
            "combat_focus": s.get("combat_focus", "balanced"),
            "caution": s.get("caution", "steady"),
            "assertiveness": clamp(s.get("assertiveness", 3), 1, 5, 3)}


def disposition_line(p):
    """A short, readable summary of how a character will fight and argue."""
    return (f"{p['name']} reads as a {p['caution']} {p['combat_focus']} fighter, "
            f"{p['max_mana']} mana, assertiveness {p['assertiveness']}/5 in council.")


def new_game(party, start="village"):
    """Build a fresh GameState for a party, with hp and mana topped up."""
    assert 1 <= len(party) <= 4, "party must be 1-4 players"
    for p in party:
        p["hp"] = p["max_hp"]
        p["mana"] = p["max_mana"]
    return GameState(location=start, party=list(party), visited={start})


def refresh_quests(gs: GameState):
    """Mark quests done once their condition holds; the only place quest status flips."""
    if "amulet" in gs.inventory:
        gs.quests["retrieve_amulet"] = "done"
    if "guardian" in gs.defeated:
        gs.quests["slay_guardian"] = "done"
