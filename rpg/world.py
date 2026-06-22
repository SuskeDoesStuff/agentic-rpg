"""The world as plain data, plus the loader that compiles it into a graph.

``WORLD`` is a dict; :func:`build_world` turns it into a NetworkX graph and checks
referential integrity (no exit to a missing room, no unknown enemy) and that every
room is reachable from the start, so typos fail loudly at load. Gates
("action:target" -> required item) are data here too, so locked paths are not
hardcoded logic. The graph (`G`), `GATES`, and `START` are immutable for a run;
the helpers that need inventory or defeated enemies take a :class:`GameState`.
"""
from __future__ import annotations

import networkx as nx

from .state import GameState

WORLD = {
    "start": "village",
    "rooms": {
        "village": {"desc": "A worn village square ringed by timber houses.",
                    "exits": ["market", "forest", "bridge"], "npcs": ["elder"]},
        "market":  {"desc": "A cramped market stall hung with lanterns and trinkets.",
                    "exits": ["village"], "items": ["torch", "potion"], "npcs": ["merchant"]},
        "forest":  {"desc": "A dim forest, roots crawling underfoot.",
                    "exits": ["village", "cave"]},
        "cave":    {"desc": "A damp cave mouth, low growling echoing within.",
                    "exits": ["forest"], "items": ["key"], "enemy": "wolf"},
        "bridge":  {"desc": "A mossy stone bridge over a black river.",
                    "exits": ["village", "ruins"]},
        "ruins":   {"desc": "Toppled columns around a sealed stone shrine.",
                    "exits": ["bridge", "crypt"], "items": ["amulet"]},
        "crypt":   {"desc": "A pitch-dark crypt, air thick with old dust.",
                    "exits": ["ruins"], "enemy": "guardian"},
    },
    "enemies": {
        "wolf":     {"hp": 14, "attack": 4},
        "guardian": {"hp": 26, "attack": 7},
    },
    "npcs": {
        "elder":    {"role": "the aged village elder, calm and a little cryptic", "knows": "quests"},
        "merchant": {"role": "a sharp-eyed trader minding the market stall", "knows": "wares"},
    },
    # "action:target" -> the inventory item required, plus the in-character refusal line
    "gates": {
        "move:crypt":  {"need": "torch", "reason": "The crypt is pitch dark; you need a light to enter."},
        "take:amulet": {"need": "key",   "reason": "The amulet sits in a locked shrine; you need a key."},
    },
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
    return {"room": room, "room_desc": G.nodes[room].get("desc", ""),
            "exits": exits(room), "items_here": items_in_room(gs, room),
            "npcs_here": npcs_at(room), "inventory": list(gs.inventory),
            "quests": dict(gs.quests)}
