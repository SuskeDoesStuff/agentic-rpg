"""The engine: movement authority, the local-work gate, and a full scripted run."""
from __future__ import annotations

from rpg import engine, players
from tests.conftest import run_gen

DEMO_ACTIONS = [
    "go to the market", "take the torch", "go back to the village",
    "head into the forest", "enter the cave", "take the key",
    "go back to the forest", "go to the village", "cross the bridge",
    "go to the ruins", "take the amulet", "go to the crypt",
]


def _agent(name, **stats):
    base = {"name": name.lower(), "max_hp": 24, "attack": 9}
    base.update(stats)
    return players.make_player(name, "an operative", "sharp", is_agent=True, stats=base)


def _human(name, **stats):
    base = {"name": name.lower(), "max_hp": 24, "attack": 9}
    base.update(stats)
    return players.make_player(name, "a warrior", "steady", is_agent=False, stats=base)


def test_movement_authority_human_leads():
    gs = players.new_game([_human("Suske"), _agent("Exys")])
    assert engine.movement_allowed(gs, gs.party[0]) is True   # human may move
    assert engine.movement_allowed(gs, gs.party[1]) is False  # agent follows the human
    gs.round_moved = True
    assert engine.movement_allowed(gs, gs.party[0]) is False  # one move per round


def test_movement_authority_all_agents():
    gs = players.new_game([_agent("A"), _agent("B")])
    assert engine.movement_allowed(gs, gs.party[0]) is True   # first agent leads
    gs.round_moved = True
    assert engine.movement_allowed(gs, gs.party[1]) is False  # the rest hold


def test_local_work_gate():
    gs = players.new_game([_agent("A")])
    gs.location = "village"
    assert engine.room_has_local_work(gs) is True  # the elder is here
    gs.location = "bridge"
    gs.inventory = []
    assert engine.room_has_local_work(gs) is False  # empty corridor


def test_full_scripted_run_completes():
    gs = players.new_game([
        _agent("Borin", max_hp=30, attack=8, combat_focus="balanced"),
        _agent("Sable", max_hp=16, attack=11, max_mana=12, combat_focus="offense"),
    ])
    gs.say = False
    gs.banter = False
    gs.scripted_actions = list(DEMO_ACTIONS)
    gs.scripted_battle = ["attack", "cast firebolt"] + ["attack"] * 30
    while gs.scripted_actions:
        p = gs.party[gs.turn % len(gs.party)]
        run_gen(engine.take_turn(gs, p))
    assert gs.location == "crypt"
    assert gs.quests_done()
    assert "wolf" in gs.defeated and "guardian" in gs.defeated
    assert all(0 <= p["hp"] <= p["max_hp"] and 0 <= p["mana"] <= p["max_mana"] for p in gs.party)


def test_second_flee_from_same_enemy_is_a_stalemate():
    gs = players.new_game([_agent("Solo", max_hp=18, attack=10)])
    gs.inventory = ["torch"]  # lit, so entry is never the blocker here
    gs.scripted_battle = ["flee"]
    gs.location = "crypt"
    assert run_gen(engine.arrive(gs, "ruins", mover=None)) == "ok"  # first flee is allowed
    assert gs.flee_counts["guardian"] == 1
    gs.scripted_battle = ["flee"]
    gs.location = "crypt"
    assert run_gen(engine.arrive(gs, "ruins", mover=None)) == "stalemate"  # second flee ends it
    assert gs.flee_counts["guardian"] == 2
