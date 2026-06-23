"""Agent reasoning: the quest-gated compass, navigation options, and the vote.

The compass only points at objectives of quests the party has actually taken, and
only routes once it has discovered where the target is. The vote is pure code.
"""
from __future__ import annotations

from rpg import agents, quests


def test_compass_names_nothing_but_explores_with_no_quest(game):
    h = agents.heading(game)
    assert h["target"] is None                 # no active quest, nothing to name
    assert h["go_to"] in agents.exits(game.location)  # but it still heads for an unvisited frontier


def test_heading_points_at_the_first_step_once_a_quest_is_taken(game):
    quests.talk(game, "elder")               # acquire the two quests, learn key/amulet/guardian
    h = agents.heading(game)
    assert h["target"] == "the key"
    assert h["go_to"] == "forest"            # village -> forest -> cave, first hop


def test_unknown_target_stops_the_route(game):
    quests.talk(game, "elder")
    for obj in ("key", "amulet"):
        game.inventory.append(obj)
    h = agents.heading(game)                  # the torch's place is the merchant's slice, not yet learned
    assert h["target"] == "the torch" and h["known"] is False
    assert "torch" not in " ".join(quests.known_goals(game))


def test_route_runs_once_every_leg_is_known(game):
    quests.talk(game, "elder")
    quests.talk(game, "merchant")             # now the torch location is known too
    for obj in ("key", "amulet", "torch"):
        game.inventory.append(obj)
    h = agents.heading(game)
    assert h["target"] == "the guardian"
    assert h["go_to"] == "bridge"             # village -> bridge -> ruins -> crypt, first hop


def test_stay_is_offered_only_while_an_npc_has_leads(game):
    assert "stay" in agents.navigation_options(game)   # the elder has quests to give
    quests.talk(game, "elder")
    opts = agents.navigation_options(game)
    assert "stay" not in opts                            # elder tapped, nothing else here
    assert set(opts) == {"market", "forest", "bridge", "tavern"}


def test_resolve_proposals_weighted_vote():
    assert agents.resolve_proposals([("A", "cave", "", 2), ("B", "market", "", 4)]) == "market"
    assert agents.resolve_proposals([("A", "cave", "", 3), ("B", "market", "", 3)], "cave") == "cave"
    assert agents.resolve_proposals([]) == "stay"
    assert agents.resolve_proposals([("A", "market", "", 2)], "forest") == "forest"
    assert agents.resolve_proposals([("A", "market", "", 3), ("B", "market", "", 3)], "forest") == "market"
