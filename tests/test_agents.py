"""Agent navigation: compass, discovery gating, and the weighted negotiation."""
from __future__ import annotations

from rpg import agents


def test_objective_chain(game):
    assert agents.next_objective(game) == ("the key", "cave")
    game.inventory = ["key"]
    assert agents.next_objective(game) == ("the amulet", "ruins")
    game.inventory = ["key", "amulet"]
    assert agents.next_objective(game) == ("the torch", "market")
    game.inventory = ["key", "amulet", "torch"]
    assert agents.next_objective(game) == ("the guardian", "crypt")


def test_compass_is_gated_on_discovery(game):
    h = agents.heading(game)
    assert h["known"] is False and h["go_to"] is None  # no intel, location unknown
    assert all("cave" not in g for g in agents.known_goals(game))  # goals stay vague
    game.intel = True
    h = agents.heading(game)
    assert h["known"] is True and h["go_to"] == "forest"  # intel unlocks the route
    assert any("cave" in g for g in agents.known_goals(game))  # and the full hint


def test_visiting_unlocks_route_without_intel(game):
    game.visited = {"village", "forest", "cave"}
    h = agents.heading(game)
    assert h["known"] is True and h["go_to"] == "forest"


def test_resolve_proposals_weighting_and_compass():
    assert agents.resolve_proposals([("A", "cave", "", 2), ("B", "market", "", 4)]) == "market"
    assert agents.resolve_proposals([("A", "cave", "", 3), ("B", "market", "", 3)], "cave") == "cave"
    assert agents.resolve_proposals([]) == "stay"
    # the compass baseline holds off a lone weak dissenter, but a real consensus overrides it
    assert agents.resolve_proposals([("A", "market", "", 2)], "forest") == "forest"
    assert agents.resolve_proposals([("A", "market", "", 3), ("B", "market", "", 3)], "forest") == "market"
