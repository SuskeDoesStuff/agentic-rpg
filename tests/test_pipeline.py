"""The per-turn pipeline must enforce world consistency in code."""
from __future__ import annotations

from rpg import pipeline


def test_illegal_actions_are_no_ops(game):
    before = (game.location, tuple(game.inventory))
    for bad in ["take the legendary sword", "teleport to the crypt", "fly to the moon"]:
        pipeline.run_turn(game, bad, "Borin")
    assert (game.location, tuple(game.inventory)) == before


def test_legal_move_and_take(game):
    pipeline.run_turn(game, "go to the market", "Borin")
    assert game.location == "market"
    pipeline.run_turn(game, "take the torch", "Borin")
    assert "torch" in game.inventory


def test_gate_blocks_locked_take(game):
    # standing in ruins without the key, the amulet cannot be taken
    game.location = "ruins"
    pipeline.run_turn(game, "take the amulet", "Borin")
    assert "amulet" not in game.inventory


def test_gate_blocks_dark_room(game):
    game.location = "ruins"  # adjacent to crypt
    pipeline.run_turn(game, "go to the crypt", "Borin")
    assert game.location == "ruins"  # no torch -> refused
    game.inventory.append("torch")
    pipeline.run_turn(game, "go to the crypt", "Borin")
    assert game.location == "crypt"


def test_go_back_resolves_previous_room(game):
    pipeline.run_turn(game, "go to the market", "Borin")
    pipeline.run_turn(game, "go back", "Borin")
    assert game.location == "village"


def test_bumping_a_gate_teaches_its_fact(game):
    game.location = "ruins"  # at the crypt's door, no torch
    assert "torch" not in game.facts
    pipeline.run_turn(game, "go to the crypt", "Borin")
    assert game.location == "ruins"  # turned back
    # the wall teaches both what it guards and where to get the light it demands
    assert "guardian" in game.facts and "torch" in game.facts
