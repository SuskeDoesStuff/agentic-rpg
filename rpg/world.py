"""The world as plain data, plus the loader that compiles it into a graph.

``WORLD`` is a dict; :func:`build_world` turns it into a NetworkX graph and checks
referential integrity (no exit to a missing room, no unknown enemy) and that every
room is reachable from the start, so typos fail loudly at load. Gates
("action:target" -> required item) are data here too, so locked paths are not
hardcoded logic.

Quests are data as well. ``OBJECTIVES`` declares each fetch/kill goal (its kind,
who reveals where it is, and the lead the party learns) and ``QUESTS`` groups
objectives under a title and a giver NPC. The quest *logic* lives entirely in
``quests.py``; nothing here knows how a quest is acquired or completed. To add a
quest, location, enemy, or NPC, edit only this file.
"""

from __future__ import annotations

import networkx as nx

from .state import GameState

WORLD = {
    "start": "village",
    "rooms": {
        "village": {
            "desc": "A worn village square ringed by timber houses.",
            "exits": ["market", "forest", "bridge", "tavern"],
            "npcs": ["elder"],
        },
        "market": {
            "desc": "A cramped market stall hung with lanterns and trinkets.",
            "exits": ["village"],
            "items": ["torch", "potion"],
            "npcs": ["merchant"],
        },
        "tavern": {
            "desc": "A low, smoky taproom thick with the smell of ale.",
            "exits": ["village"],
            "npcs": ["barkeep"],
        },
        "forest": {"desc": "A dim forest, roots crawling underfoot.", "exits": ["village", "cave", "grove"]},
        "cave": {
            "desc": "A damp cave mouth, low growling echoing within.",
            "exits": ["forest"],
            "items": ["key"],
            "enemy": "wolf",
        },
        "grove": {
            "desc": "A trampled grove of broken saplings and churned earth.",
            "exits": ["forest"],
            "enemy": "bear",
        },
        "bridge": {"desc": "A mossy stone bridge over a black river.", "exits": ["village", "ruins"]},
        "ruins": {
            "desc": "Toppled columns around a sealed stone shrine.",
            "exits": ["bridge", "crypt", "chapel"],
            "items": ["amulet"],
        },
        "crypt": {"desc": "A pitch-dark crypt, air thick with old dust.", "exits": ["ruins"], "enemy": "guardian"},
        "chapel": {
            "desc": "A roofless chapel of cracked pews and a cold altar.",
            "exits": ["ruins", "sanctum"],
            "items": ["ward"],
            "npcs": ["priest"],
        },
        "sanctum": {
            "desc": "A sealed inner sanctum where the air does not move.",
            "exits": ["chapel"],
            "enemy": "wraith",
        },
    },
    "enemies": {
        "wolf": {"hp": 14, "attack": 4},
        "bear": {"hp": 20, "attack": 5},
        "guardian": {"hp": 26, "attack": 7},
        "wraith": {"hp": 22, "attack": 6},
    },
    "npcs": {
        "elder": {
            "role": "the aged village elder, calm and a little cryptic",
            "voice": "an old keeper of this land's lore who lays out the tasks that lie ahead",
        },
        "merchant": {
            "role": "a sharp-eyed trader minding the market stall",
            "voice": "a market trader who speaks only of the wares on the table and knows nothing of "
            "far places, quests, or where anything else lies",
        },
        "barkeep": {
            "role": "a broad tavern-keeper with an ear for every rumour",
            "voice": "a tavern-keeper who deals in gossip and bounties on the beasts that trouble the roads",
        },
        "priest": {
            "role": "a hooded priest tending the chapel's old rites",
            "voice": "a quiet priest who speaks of wards, restless dead, and the sanctum's broken seal",
        },
    },
    # "action:target" -> the inventory item required, plus the in-character refusal line
    "gates": {
        "move:crypt": {
            "need": "torch",
            "reason": "The crypt is pitch dark; you need a torch, such as the one sold in the market.",
        },
        "take:amulet": {
            "need": "key",
            "reason": "The amulet sits in a locked shrine; you need the key from the cave past the forest.",
        },
        "move:sanctum": {
            "need": "ward",
            "reason": "The sanctum's seal repels you; you need a ward, the charm kept in the chapel.",
        },
    },
}

# A quest objective: an item to fetch or an enemy to kill, who reveals where it is, and the lead the party learns.
# The room is derived from the graph (node_room), never duplicated here.
OBJECTIVES = {
    "key": {
        "kind": "fetch",
        "reveal_by": "elder",
        "lead": "a key lies in the cave past the forest, with a wolf set to guard it",
    },
    "amulet": {
        "kind": "fetch",
        "reveal_by": "elder",
        "lead": "the amulet rests in the ruins, sealed behind a locked shrine",
    },
    "torch": {
        "kind": "fetch",
        "reveal_by": "merchant",
        "lead": "a torch waits in the market, a light for the dark places",
    },
    "ward": {"kind": "fetch", "reveal_by": "priest", "lead": "a warding charm sits in the chapel beside the ruins"},
    "guardian": {
        "kind": "kill",
        "reveal_by": "elder",
        "lead": "a guardian holds the crypt, and the crypt is pitch dark",
    },
    "bear": {"kind": "kill", "reveal_by": "barkeep", "lead": "a great bear prowls the grove off the forest"},
    "wraith": {"kind": "kill", "reveal_by": "priest", "lead": "a wraith haunts the sanctum, sealed beyond the chapel"},
}

# A quest: a titled goal, the NPC who grants it on a talk, and its objectives in order (earlier ones gate later ones).
QUESTS = {
    "amulet_hunt": {"title": "Recover the Amulet", "giver": "elder", "steps": ["key", "amulet"]},
    "kill_guardian": {"title": "Kill the Guardian", "giver": "elder", "steps": ["torch", "guardian"]},
    "bear_hunt": {"title": "Hunt the Grove Bear", "giver": "barkeep", "steps": ["bear"]},
    "cleanse_sanctum": {"title": "Cleanse the Sanctum", "giver": "priest", "steps": ["ward", "wraith"]},
}


def build_world(spec):
    """Compile the declarative spec into a graph, validating references and reachability."""
    g = nx.DiGraph()
    rooms = spec["rooms"]
    for name, r in rooms.items():
        g.add_node(name, type="room", desc=r["desc"])
    for name, r in rooms.items():
        for dest in r.get("exits", []):
            assert dest in rooms, f"room '{name}' exits to unknown room '{dest}'"
            g.add_edge(name, dest, type="exit")
            g.add_edge(dest, name, type="exit")  # bidirectional
        for it in r.get("items", []):
            g.add_node(it, type="item")
            g.add_edge(it, name, type="in")
        for npc in r.get("npcs", []):
            g.add_node(npc, type="npc")
            g.add_edge(npc, name, type="at")
        if "enemy" in r:
            e = r["enemy"]
            assert e in spec["enemies"], f"room '{name}' has unknown enemy '{e}'"
            st = spec["enemies"][e]
            g.add_node(e, type="enemy", hp=st["hp"], attack=st["attack"])
            g.add_edge(e, name, type="guards")
    for k in spec.get("gates", {}):
        act, _, tgt = k.partition(":")
        assert act and tgt, f"gate key '{k}' must be 'action:target'"
    start = spec.get("start", next(iter(rooms)))
    reachable = nx.descendants(g, start) | {start}
    stranded = [r for r in rooms if r not in reachable]
    assert not stranded, f"rooms unreachable from '{start}': {stranded}"
    return g


G = build_world(WORLD)
GATES = WORLD["gates"]
START = WORLD.get("start", "village")


def node_room(name):
    """The room a non-room node (item, npc, enemy) lives in."""
    for _, r, e in G.out_edges(name, data=True):
        if e["type"] in ("in", "guards", "at"):
            return r
    return None


def _validate_quests():
    """Fail loudly at load if the quest tables reference anything the graph does not contain."""
    npcs = WORLD["npcs"]
    for name, o in OBJECTIVES.items():
        assert name in G and G.nodes[name]["type"] in ("item", "enemy"), f"objective '{name}' is not an item or enemy"
        assert (o["kind"] == "kill") == (G.nodes[name]["type"] == "enemy"), (
            f"objective '{name}' kind mismatches its node"
        )
        assert o["reveal_by"] in npcs, f"objective '{name}' revealed by unknown npc '{o['reveal_by']}'"
        assert node_room(name), f"objective '{name}' is not placed in any room"
    for qid, q in QUESTS.items():
        assert q["giver"] in npcs, f"quest '{qid}' given by unknown npc '{q['giver']}'"
        for step in q["steps"]:
            assert step in OBJECTIVES, f"quest '{qid}' names unknown objective '{step}'"


_validate_quests()


def exits(room):
    """Rooms reachable in one step from ``room``."""
    return [v for _, v, e in G.out_edges(room, data=True) if e.get("type") == "exit"]


def items_in_room(gs: GameState, room):
    """Items physically in ``room`` that the party is not already carrying."""
    here = [u for u, _, e in G.in_edges(room, data=True) if e.get("type") == "in"]
    return [i for i in here if i not in gs.inventory]


def npcs_at(room):
    """NPCs present in ``room``."""
    return [u for u, _, e in G.in_edges(room, data=True) if e.get("type") == "at"]


def enemy_in_room(gs: GameState, room):
    """A live (undefeated) enemy guarding ``room``, or None."""
    for u, _, e in G.in_edges(room, data=True):
        if e.get("type") == "guards" and u not in gs.defeated:
            return u
    return None


def world_context(gs: GameState, room):
    """A snapshot of what is true here, the only world knowledge handed to the model."""
    return {
        "room": room,
        "room_desc": G.nodes[room].get("desc", ""),
        "exits": exits(room),
        "items_here": items_in_room(gs, room),
        "npcs_here": npcs_at(room),
        "inventory": list(gs.inventory),
        "quests": dict(gs.quests),
    }
