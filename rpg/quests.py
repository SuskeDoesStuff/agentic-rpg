"""Quest logic, derived entirely from the declarative tables in ``world.py``.

No quest data lives here, only the rules that read ``OBJECTIVES`` and ``QUESTS``
against a :class:`GameState`:

- what the party still needs (``next_objective``) and what leads it holds (``known_goals``),
- where an objective is and whether it is known or done,
- what an NPC reveals and which quests it grants when talked to (``talk``),
- when a quest finishes (``complete``) and when the whole game is won (``all_done``),
- what a bumped gate teaches (``gate_facts``).

Acquisition is "talk to the right NPC"; the engine greets every NPC in a newly
found room, so quests surface through exploration. Adding content is a pure data
edit in ``world.py`` because every rule below is generic over the tables.
"""
from __future__ import annotations

from .world import GATES, OBJECTIVES, QUESTS, node_room


def has_objective(gs, name):
    """Whether an objective is already secured: an enemy slain, or an item carried."""
    if OBJECTIVES[name]["kind"] == "kill":
        return name in gs.defeated
    return name in gs.inventory


def objective_room(name):
    """The room an objective sits in, read from the graph."""
    return node_room(name)


def knows_where(gs, name):
    """The party knows where an objective is once told of it (a fact) or once it has stood in its room."""
    return name in gs.facts or objective_room(name) in gs.visited


def active_objective_names(gs):
    """Every objective of every acquired-and-open quest, in quest order then step order."""
    names = []
    for qid, q in QUESTS.items():
        if gs.quests.get(qid) == "active":
            names.extend(q["steps"])
    return names


def next_objective(gs):
    """The next unfinished objective among active quests, as (label, room); (None, None) when none remain."""
    nxt = next((n for n in active_objective_names(gs) if not has_objective(gs, n)), None)
    return (f"the {nxt}", objective_room(nxt)) if nxt else (None, None)


def known_goals(gs):
    """The party's actionable leads: located-but-unclaimed objectives of active quests."""
    leads = [OBJECTIVES[n]["lead"] for n in active_objective_names(gs)
             if knows_where(gs, n) and not has_objective(gs, n)]
    return leads or ["no firm leads yet; seek out new places and the people who hand out tasks"]


def npc_has_more(gs, npc):
    """Whether talking to this NPC would teach or grant anything new (a lead or an unacquired quest)."""
    gives_new = any(q["giver"] == npc and qid not in gs.quests for qid, q in QUESTS.items())
    reveals_new = any(o["reveal_by"] == npc and not knows_where(gs, n) for n, o in OBJECTIVES.items())
    return gives_new or reveals_new


def talk(gs, npc):
    """Resolve a conversation: acquire the NPC's quests and reveal what it knows. Mutates ``gs``.

    Returns the titles of quests just acquired and the leads just learned, for the caller to voice and announce.
    """
    acquired = []
    for qid, q in QUESTS.items():
        if q["giver"] == npc and qid not in gs.quests:
            gs.quests[qid] = "active"
            acquired.append(q["title"])
    leads = []
    for name, o in OBJECTIVES.items():
        if o["reveal_by"] == npc and not knows_where(gs, name):
            gs.facts.add(name)
            leads.append(o["lead"])
    return {"acquired": acquired, "leads": leads}


def complete(gs):
    """Mark any active quest whose every step is secured as done; return the titles just completed."""
    finished = []
    for qid, q in QUESTS.items():
        if gs.quests.get(qid) == "active" and all(has_objective(gs, s) for s in q["steps"]):
            gs.quests[qid] = "done"
            finished.append(q["title"])
    return finished


def all_done(gs):
    """The game is won only when every quest in the world has been acquired and completed."""
    return all(gs.quests.get(qid) == "done" for qid in QUESTS)


def gate_facts(gatekey):
    """What bumping a gate teaches: where to get the item it demands, plus what waits behind it."""
    gate = GATES.get(gatekey, {})
    taught = set()
    if gate.get("need") in OBJECTIVES:
        taught.add(gate["need"])  # where to get what the gate demands, so no objective is orphaned
    action, _, target = gatekey.partition(":")
    if action == "move":
        taught |= {n for n in OBJECTIVES if objective_room(n) == target}  # what the barred room holds
    elif target in OBJECTIVES:
        taught.add(target)
    return taught
