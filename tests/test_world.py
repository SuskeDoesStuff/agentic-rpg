"""World construction, graph helpers, and player derivation."""
from __future__ import annotations

import networkx as nx
import pytest

from rpg import players, world


def test_world_builds_and_is_reachable():
    assert world.G.number_of_nodes() == 24
    assert world.G.number_of_edges() == 33
    reachable = nx.descendants(world.G, world.START) | {world.START}
    assert all(r in reachable for r in world.WORLD["rooms"])


def test_build_world_rejects_dangling_exit():
    bad = {"start": "a", "rooms": {"a": {"desc": "", "exits": ["ghost"]}}, "enemies": {}, "npcs": {}}
    with pytest.raises(AssertionError):
        world.build_world(bad)


def test_graph_helpers_read_state(game):
    assert set(world.exits("village")) == {"market", "forest", "bridge", "tavern"}
    assert set(world.items_in_room(game, "market")) == {"torch", "potion"}
    assert world.npcs_at("village") == ["elder"]
    assert world.npcs_at("chapel") == ["priest"]
    assert world.enemy_in_room(game, "grove") == "bear"
    assert world.enemy_in_room(game, "cave") == "wolf"
    game.defeated.add("wolf")
    assert world.enemy_in_room(game, "cave") is None  # a defeated enemy is gone


def test_items_in_room_hides_carried(game):
    game.inventory = ["torch"]
    assert world.items_in_room(game, "market") == ["potion"]


def test_player_derives_disposition_and_mana():
    p = players.make_player("Vesna", "a court wizard", "calm and cautious", is_agent=True,
                            stats={"name": "wizard", "max_hp": 16, "attack": 6, "max_mana": 12,
                                   "combat_focus": "offense", "caution": "cautious", "assertiveness": 2})
    assert p["combat_focus"] == "offense" and p["caution"] == "cautious"
    assert p["max_mana"] == 12 and 1 <= p["assertiveness"] <= 5


def test_stats_clamped_to_safe_range():
    p = players.make_player("X", "broken", stats={"name": "x", "max_hp": 9999, "attack": 999, "max_mana": 999})
    assert p["max_hp"] == players.HP_MAX and p["attack"] == players.ATK_MAX and p["max_mana"] == players.MANA_MAX


def test_new_game_tops_up_and_validates():
    gs = players.new_game([players.make_player("A", "x", stats={"name": "x", "max_hp": 20, "attack": 8, "max_mana": 6})])
    assert gs.party[0]["hp"] == 20 and gs.party[0]["mana"] == 6
    assert gs.location == "village" and "village" in gs.visited
    with pytest.raises(AssertionError):
        players.new_game([])
