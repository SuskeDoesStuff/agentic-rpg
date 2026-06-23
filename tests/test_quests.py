"""Quest logic: objectives, talk-acquisition, completion, and the win condition.

All of this is derived from the ``OBJECTIVES`` and ``QUESTS`` tables in world.py;
these tests pin the rules so a content edit can't silently change behaviour.
"""
from __future__ import annotations

from rpg import quests


def test_no_quests_means_no_objective(game):
    assert game.quests == {}
    assert quests.next_objective(game) == (None, None)
    assert quests.known_goals(game)[0].startswith("no firm leads")


def test_talking_to_elder_acquires_and_reveals(game):
    out = quests.talk(game, "elder")
    assert out["acquired"] == ["Recover the Amulet", "Kill the Guardian"]
    assert game.quests == {"amulet_hunt": "active", "kill_guardian": "active"}
    assert {"key", "amulet", "guardian"} <= game.facts  # the elder's slice, learned in one exchange
    assert "torch" not in game.facts                    # the torch is the merchant's slice
    assert quests.next_objective(game) == ("the key", "cave")


def test_objective_order_follows_active_quests(game):
    quests.talk(game, "elder")
    assert quests.next_objective(game) == ("the key", "cave")
    game.inventory.append("key")
    assert quests.next_objective(game) == ("the amulet", "ruins")
    game.inventory.append("amulet")
    assert quests.next_objective(game) == ("the torch", "market")
    game.inventory.append("torch")
    assert quests.next_objective(game) == ("the guardian", "crypt")
    game.defeated.add("guardian")
    assert quests.next_objective(game) == (None, None)  # both elder quests' steps are done


def test_kill_and_fetch_objectives_read_different_state(game):
    assert quests.has_objective(game, "guardian") is False
    game.defeated.add("guardian")
    assert quests.has_objective(game, "guardian") is True   # kill -> defeated set
    assert quests.has_objective(game, "key") is False
    game.inventory.append("key")
    assert quests.has_objective(game, "key") is True        # fetch -> inventory


def test_completion_flips_once_and_only_when_all_steps_done(game):
    quests.talk(game, "elder")
    game.inventory.append("key")
    assert quests.complete(game) == []          # amulet_hunt still needs the amulet
    game.inventory.append("amulet")
    assert quests.complete(game) == ["Recover the Amulet"]
    assert game.quests["amulet_hunt"] == "done"
    assert quests.complete(game) == []          # never announced twice


def test_bear_hunt_completes_on_a_single_kill(game):
    quests.talk(game, "barkeep")
    assert game.quests["bear_hunt"] == "active"
    game.defeated.add("bear")
    assert quests.complete(game) == ["Hunt the Grove Bear"]


def test_all_done_needs_every_quest_acquired_and_finished(game):
    # securing every objective is not a win until each quest has been taken from its giver
    for obj in ("key", "amulet", "torch", "ward"):
        game.inventory.append(obj)
    for foe in ("guardian", "bear", "wraith"):
        game.defeated.add(foe)
    assert quests.all_done(game) is False       # nothing acquired yet
    for npc in ("elder", "barkeep", "priest"):
        quests.talk(game, npc)
    quests.complete(game)
    assert quests.all_done(game) is True


def test_npc_goes_quiet_once_tapped(game):
    assert quests.npc_has_more(game, "elder") is True       # has quests to give
    quests.talk(game, "elder")
    assert quests.npc_has_more(game, "elder") is False      # gave them and revealed its slice
    assert quests.npc_has_more(game, "merchant") is True    # still holds the torch lead
    quests.talk(game, "merchant")
    assert quests.npc_has_more(game, "merchant") is False


def test_gate_facts_are_derived_from_the_tables(game):
    assert quests.gate_facts("move:crypt") == {"torch", "guardian"}
    assert quests.gate_facts("take:amulet") == {"key", "amulet"}
    assert quests.gate_facts("move:sanctum") == {"ward", "wraith"}
