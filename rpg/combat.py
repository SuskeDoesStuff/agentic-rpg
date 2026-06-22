"""The battle system.

Each round every fighter chooses a move (attack, cast, potion, defend, flee) and
the engine resolves it in pure code, so the model decides intent but never the
numbers. Spells cost mana, potions are a shared consumable, defending softens the
next blow, HP and mana persist across fights. A deterministic ``battle_outlook``
tells a fighter whether it is on track to win or to fall first, so survival can
override temperament. The enemy is deterministic and targets the biggest threat.
``run_battle`` is a generator: it yields System lines, requests a human's move via
NeedBattleChoice, and returns 'won', 'lost', or 'fled'.
"""
from __future__ import annotations

import json

from . import config
from .events import NeedBattleChoice, System
from .players import refresh_quests
from .schemas import CombatMove
from .speech import banter
from .world import G

POTION_HEAL, DEFEND_REDUCE = 12, 5
SPELLS = {
    "firebolt": {"cost": 3, "kind": "attack", "power": 9},
    "mend":     {"cost": 4, "kind": "heal",   "power": 12},
    "ward":     {"cost": 2, "kind": "defend", "power": 6},
}


def battle_menu(gs, p):
    """The moves legal for this fighter right now, given mana and the shared inventory."""
    opts = ["attack", "defend", "flee"]
    if "potion" in gs.inventory:
        opts.append("potion")
    opts += [f"cast {s}" for s, v in SPELLS.items() if v["cost"] <= p.get("mana", 0)]
    return opts


def parse_battle_raw(raw):
    """Map a typed or scripted battle string onto a CombatMove dict."""
    raw = (raw or "").lower().strip()
    if raw.startswith("cast"):
        return {"move": "cast", "spell": raw[4:].strip()}
    for m in ("attack", "defend", "flee", "potion"):
        if m in raw:
            return {"move": m, "spell": ""}
    return {"move": "attack", "spell": ""}


def validate_move(gs, p, mv):
    """Force the move legal: an unaffordable cast or a missing potion falls back to a swing."""
    move = (mv.get("move") or "attack").lower().strip()
    spell = (mv.get("spell") or "").lower().strip()
    if move == "cast" and (spell not in SPELLS or SPELLS[spell]["cost"] > p.get("mana", 0)):
        move, spell = "attack", ""
    if move == "potion" and "potion" not in gs.inventory:
        move = "attack"
    return {"move": move, "spell": spell}


def battle_outlook(gs, p, ehp, eatk):
    """A deterministic read on the race: are we on track to win, or to fall first?"""
    party_dps = max(1, sum(a["attack"] for a in gs.alive()))
    rounds_to_kill = -(-ehp // party_dps)  # ceil
    my_rounds_to_die = -(-p["hp"] // max(1, eatk))  # ceil, ignoring guard/heals
    healing = ("potion" in gs.inventory) or any(
        SPELLS[s]["kind"] == "heal" and SPELLS[s]["cost"] <= p.get("mana", 0) for s in SPELLS
    )
    losing = my_rounds_to_die < rounds_to_kill  # party acts first, so a tie kills the enemy before you fall
    return {"rounds_to_kill_enemy": rounds_to_kill, "rounds_until_you_fall": my_rounds_to_die,
            "you_can_heal": healing, "outlook": "losing" if losing else "winning"}


def lowest_ally(gs):
    """The living party member in the most trouble, by HP fraction."""
    return min(gs.alive(), key=lambda a: a["hp"] / max(1, a["max_hp"]))


def battle_choice(gs, p, enemy, ehp, eatk):
    """Pick a move: scripted queue first, else the agent reasons, else request the human's input."""
    if gs.scripted_battle:
        return validate_move(gs, p, parse_battle_raw(gs.scripted_battle.pop(0)))
    menu = battle_menu(gs, p)
    if p["is_agent"]:
        sysm = (f"You are {p['name']}, a {p['caution']} {p['combat_focus']} fighter. Choose ONE move from the "
                "options that fits your nature and this moment. Offense favors attack or an attack spell; support "
                "favors mending the most-hurt ally; the reckless press the attack. But survival overrides "
                "temperament: if 'outlook' is losing and your HP is low (below about half your max), heal THIS TURN "
                "if you can, since defending only delays death while a heal reverses it. If you are losing but still "
                "near full HP, a heal would be wasted and the foe is simply too strong, so flee rather than trade "
                "blows to your death. Never spend a potion or heal spell while your HP is already high. Spend mana "
                "and potions deliberately, they do not refill mid-fight.")
        usr = json.dumps({"options": menu, "your_hp": [p["hp"], p["max_hp"]], "your_mana": p.get("mana", 0),
                          "party": [{"name": a["name"], "hp": a["hp"], "max_hp": a["max_hp"]} for a in gs.alive()],
                          "enemy": {"name": enemy, "hp": ehp, "attack": eatk},
                          "outlook": battle_outlook(gs, p, ehp, eatk)})
        return validate_move(gs, p, config.work_struct(CombatMove, [("system", sysm), ("human", usr)]))
    raw = yield NeedBattleChoice(p["name"], menu)
    return validate_move(gs, p, parse_battle_raw(raw))


def apply_move(gs, p, mv, enemy, ehp):
    """Resolve one chosen move deterministically. Yields System lines, returns (enemy_hp, fled)."""
    move, spell = mv["move"], mv["spell"]
    if move == "flee":
        return ehp, True
    if move == "defend":
        p["guard"] = DEFEND_REDUCE
        yield System(f"{p['name']} braces behind a guard.")
        return ehp, False
    if move == "potion":
        gs.inventory.remove("potion")
        heal = min(POTION_HEAL, p["max_hp"] - p["hp"])
        p["hp"] += heal
        yield System(f"{p['name']} drinks a potion, recovering {heal} hp (hp {p['hp']}).")
        return ehp, False
    if move == "cast":
        v = SPELLS[spell]
        p["mana"] = max(0, p.get("mana", 0) - v["cost"])
        if v["kind"] == "attack":
            ehp = max(0, ehp - v["power"])
            yield System(f"{p['name']} casts {spell}, searing the {enemy} for {v['power']} (enemy hp {ehp}).")
        elif v["kind"] == "heal":
            t = lowest_ally(gs)
            heal = min(v["power"], t["max_hp"] - t["hp"])
            t["hp"] += heal
            yield System(f"{p['name']} casts {spell}, mending {t['name']} for {heal} (hp {t['hp']}).")
        else:
            p["guard"] = v["power"]
            yield System(f"{p['name']} casts {spell}, weaving a ward.")
        return ehp, False
    ehp = max(0, ehp - p["attack"])
    yield System(f"{p['name']} hits the {enemy} for {p['attack']} (enemy hp {ehp}).")
    return ehp, False


def run_battle(gs, enemy, prev_room):
    """Turn-based fight. Yields System lines and battle prompts; returns 'won', 'lost', or 'fled'."""
    ehp, eatk = G.nodes[enemy]["hp"], G.nodes[enemy]["attack"]
    yield System(f"-- a {enemy} attacks! (hp {ehp}) --")
    gs.remember(f"(a {enemy} attacks the party)")
    yield from banter(gs, f"a {enemy} just attacked the party")
    for p in gs.party:
        p["guard"] = 0
    while ehp > 0 and gs.alive():
        for p in list(gs.alive()):
            mv = yield from battle_choice(gs, p, enemy, ehp, eatk)
            ehp, fled = yield from apply_move(gs, p, mv, enemy, ehp)
            if fled:
                gs.location = prev_room
                gs.flee_counts[enemy] = gs.flee_counts.get(enemy, 0) + 1
                yield System(f"{p['name']} calls the retreat; the party flees to the {prev_room}.")
                gs.remember(f"(the party fled from the {enemy})")
                return "fled"
            if ehp <= 0:
                break
        if ehp > 0 and gs.alive():
            tgt = max(gs.alive(), key=lambda a: a["attack"])  # the enemy goes for the biggest threat
            dmg = max(0, eatk - tgt.get("guard", 0))
            tgt["guard"] = 0
            tgt["hp"] = max(0, tgt["hp"] - dmg)
            yield System(f"the {enemy} strikes {tgt['name']} for {dmg} (hp {tgt['hp']}).")
    if ehp <= 0:
        gs.defeated.add(enemy)
        refresh_quests(gs)
        yield System(f"-- the {enemy} is defeated! --")
        gs.remember(f"(the party defeated the {enemy})")
        yield from banter(gs, f"the party just defeated the {enemy}")
        return "won"
    yield System("-- the party has fallen --")
    return "lost"
