"""Combat: agents and humans choose, the engine resolves in pure code."""
from __future__ import annotations

from rpg import combat, players
from tests.conftest import run_gen


def _invariants_hold(gs):
    return all(0 <= p["hp"] <= p["max_hp"] and 0 <= p["mana"] <= p["max_mana"] for p in gs.party)


def test_scripted_battle_is_won_deterministically():
    gs = players.new_game([
        players.make_player("Borin", "sellsword", is_agent=True, stats={"name": "s", "max_hp": 30, "attack": 8}),
        players.make_player("Sable", "mage", is_agent=True, stats={"name": "m", "max_hp": 16, "attack": 11}),
    ])
    gs.location = "forest"
    gs.scripted_battle = ["attack"] * 20
    out = run_gen(combat.run_battle(gs, "wolf", "forest"))
    assert out == "won" and "wolf" in gs.defeated and _invariants_hold(gs)


def test_magic_potion_and_mana():
    gs = players.new_game([players.make_player("Vesna", "wizard", is_agent=True,
                          stats={"name": "w", "max_hp": 16, "attack": 6, "max_mana": 12, "combat_focus": "offense"})])
    gs.location = "forest"
    gs.inventory = ["potion"]
    m0 = gs.party[0]["mana"]
    gs.scripted_battle = ["cast firebolt", "potion", "cast firebolt", "attack", "attack", "attack"]
    out = run_gen(combat.run_battle(gs, "wolf", "forest"))
    assert out == "won"
    assert m0 - gs.party[0]["mana"] >= 3  # mana was spent on casts
    assert "potion" not in gs.inventory  # the potion was consumed
    assert _invariants_hold(gs)


def test_unaffordable_cast_falls_back_to_attack():
    gs = players.new_game([players.make_player("X", "fighter", is_agent=True,
                          stats={"name": "x", "max_hp": 20, "attack": 8, "max_mana": 1})])
    mv = combat.validate_move(gs, gs.party[0], {"move": "cast", "spell": "firebolt"})
    assert mv["move"] == "attack"


def test_battle_menu_respects_resources():
    gs = players.new_game([players.make_player("X", "mage", is_agent=True,
                          stats={"name": "x", "max_hp": 16, "attack": 6, "max_mana": 5})])
    gs.inventory = []
    menu = combat.battle_menu(gs, gs.party[0])
    assert "potion" not in menu and any(m.startswith("cast") for m in menu)
    gs.party[0]["mana"] = 0
    assert not any(m.startswith("cast") for m in combat.battle_menu(gs, gs.party[0]))


def test_outlook_reads_the_race():
    gs = players.new_game([players.make_player("Frail", "mage", is_agent=True,
                          stats={"name": "m", "max_hp": 14, "attack": 10, "max_mana": 2})])
    gs.inventory = []
    gs.party[0]["hp"] = 7
    assert combat.battle_outlook(gs, gs.party[0], 16, 7)["outlook"] == "losing"
    gs.inventory = ["potion"]
    assert combat.battle_outlook(gs, gs.party[0], 16, 7)["you_can_heal"] is True

    tank = players.new_game([players.make_player("Tank", "bruiser", is_agent=True,
                            stats={"name": "t", "max_hp": 30, "attack": 10})])
    assert combat.battle_outlook(tank, tank.party[0], 10, 4)["outlook"] == "winning"
