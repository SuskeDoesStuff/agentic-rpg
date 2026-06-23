"""Agent navigation: the canonical chain, facts-based discovery, and the weighted vote."""
from __future__ import annotations

from rpg import agents, speech


def test_objective_chain_respects_dependencies(game):
    assert agents.next_objective(game) == ("the key", "cave")
    game.inventory = ["key"]
    assert agents.next_objective(game) == ("the amulet", "ruins")
    game.inventory = ["key", "amulet"]
    assert agents.next_objective(game) == ("the torch", "market")
    game.inventory = ["key", "amulet", "torch"]
    assert agents.next_objective(game) == ("the guardian", "crypt")


def test_heading_stays_dark_until_the_fact_is_known(game):
    h = agents.heading(game)
    assert h["known"] is False and h["go_to"] is None  # the cave's whereabouts are unknown
    assert "cave" not in " ".join(agents.known_goals(game))
    game.facts.add("key")  # the elder or a visit revealed where the key lies
    h = agents.heading(game)
    assert h["known"] is True and h["go_to"] == "forest"  # the lead unlocks the route
    assert any("cave" in g for g in agents.known_goals(game))


def test_visiting_a_room_counts_as_knowing_it(game):
    game.visited = {"village", "forest", "cave"}  # been to the cave, so its contents are known
    h = agents.heading(game)
    assert h["known"] is True and h["go_to"] == "forest"


def test_npc_reveal_is_sliced_and_advances_once_told(game):
    assert agents.npc_reveal(game, "merchant") == "torch"  # the merchant only knows its stall
    assert agents.npc_reveal(game, "elder") == "key"       # the elder's first unknown leg
    game.facts.add("key")
    assert agents.npc_reveal(game, "elder") == "amulet"     # told the key, it advances
    game.facts.add("amulet")
    assert agents.npc_reveal(game, "elder") == "guardian"
    game.facts.add("guardian")
    assert agents.npc_reveal(game, "elder") is None         # fully tapped, nothing new


def test_asking_an_npc_grants_the_fact(game):
    assert "torch" not in game.facts
    speech.npc_reply(game, "merchant", "what do you sell?", "Borin")
    assert "torch" in game.facts  # the merchant's lead is now the party's knowledge


def test_navigation_drops_stay_when_a_room_is_tapped_out(game):
    assert "stay" in agents.navigation_options(game)  # the elder still has leads to give
    game.facts |= {"key", "amulet", "guardian"}       # elder fully tapped, no items, no enemy
    opts = agents.navigation_options(game)
    assert "stay" not in opts and set(opts) == {"market", "forest", "bridge"}


def test_heading_lists_unexplored_paths_when_lost(game):
    h = agents.heading(game)  # no leads yet, standing in the village
    assert h["known"] is False
    assert set(h["unexplored"]) == {"market", "forest", "bridge"}


def test_discovery_walk_unlocks_the_whole_chain(game):
    assert agents.heading(game)["known"] is False
    game.facts.add(agents.npc_reveal(game, "elder"))     # key
    game.facts.add(agents.npc_reveal(game, "elder"))     # amulet
    game.facts.add(agents.npc_reveal(game, "elder"))     # guardian
    game.facts.add(agents.npc_reveal(game, "merchant"))  # torch
    assert agents.npc_reveal(game, "elder") is None and agents.npc_reveal(game, "merchant") is None
    game.location = "ruins"
    game.inventory = ["key", "amulet", "torch"]
    assert agents.heading(game)["go_to"] == "crypt"  # every leg known, the route runs to the crypt


def test_resolve_proposals_weighting_and_compass():
    assert agents.resolve_proposals([("A", "cave", "", 2), ("B", "market", "", 4)]) == "market"
    assert agents.resolve_proposals([("A", "cave", "", 3), ("B", "market", "", 3)], "cave") == "cave"
    assert agents.resolve_proposals([]) == "stay"
    # the compass baseline holds off a lone weak dissenter, but a real consensus overrides it
    assert agents.resolve_proposals([("A", "market", "", 2)], "forest") == "forest"
    assert agents.resolve_proposals([("A", "market", "", 3), ("B", "market", "", 3)], "forest") == "market"
