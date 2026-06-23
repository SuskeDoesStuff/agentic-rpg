"""Agent reasoning: the compass, discovery-gated goals, per-turn decisions, and the
no-human navigation vote.

The graph computes the correct next objective and route; agents read it but only
learn where things are once the party discovers them. With no human present, each
agent argues for a destination and a deterministic, assertiveness-weighted vote
resolves it, with the compass as a baseline voice so a confident-but-wrong
consensus needs real support to override the proven route.
"""
from __future__ import annotations

import json
from collections import deque

import networkx as nx

from . import config
from .events import Argument
from .players import clamp
from .quests import known_goals, knows_where, next_objective, npc_has_more
from .schemas import AgentTurn, Proposal
from .world import GATES, G, enemy_in_room, exits, items_in_room, npcs_at, world_context


def roster(gs, player):
    """How the other party members read to one agent, tagged human or ally."""
    out = []
    for p in gs.party:
        if p is player:
            continue
        tag = "the human player" if not p["is_agent"] else "an ally"
        out.append(f"{p['name']} the {p['class_name']} ({tag})")
    return ", ".join(out) or "no one else"


def _blocked(gs, room):
    """Whether a move-gate currently bars entry to ``room`` for want of the required item."""
    gate = GATES.get(f"move:{room}")
    return bool(gate) and gate["need"] not in gs.inventory


def explore_hop(gs):
    """First step toward the nearest unvisited room the party may actually enter, for blind exploration.

    BFS runs over known (visited) rooms only and stops at the first unvisited, ungated-or-unlocked frontier;
    blocked rooms (a dark crypt, a sealed sanctum) are left to the compass, which sequences their key item first.
    """
    q = deque([(gs.location, None)])
    seen = {gs.location}
    while q:
        room, first = q.popleft()
        for nxt in exits(room):
            if nxt in seen:
                continue
            seen.add(nxt)
            hop = nxt if first is None else first
            if nxt not in gs.visited:
                if not _blocked(gs, nxt):
                    return hop  # nearest reachable frontier
                continue        # a barred frontier: do not route through it
            q.append((nxt, hop))  # a known room: keep searching beyond it
    return None


def heading(gs):
    """Objective and the next room toward it; when there is none or its place is unknown, steer to the frontier."""
    obj, dest = next_objective(gs)
    unexplored = [e for e in exits(gs.location) if e not in gs.visited]
    if not dest:  # nothing active to chase: explore for the people who hand out the rest of the tasks
        return {"target": None, "go_to": explore_hop(gs), "known": True, "unexplored": unexplored}
    known = knows_where(gs, obj.removeprefix("the "))
    if dest == gs.location:
        return {"target": obj, "go_to": None, "known": True, "unexplored": unexplored}
    if not known:  # we know it exists but not where: wander the frontier until we are told or stumble on it
        return {"target": obj, "go_to": explore_hop(gs), "known": False, "unexplored": unexplored}
    try:
        path = nx.shortest_path(G, gs.location, dest)
        return {"target": obj, "go_to": (path[1] if len(path) > 1 else None), "known": True, "unexplored": unexplored}
    except Exception:
        return {"target": obj, "go_to": None, "known": True, "unexplored": unexplored}


def navigation_options(gs):
    """Rooms the party may move to, plus 'stay' only when something here is worth staying for."""
    opts = exits(gs.location)
    worth_staying = bool(items_in_room(gs, gs.location) or enemy_in_room(gs, gs.location)
                         or any(npc_has_more(gs, n) for n in npcs_at(gs.location)))
    return opts + (["stay"] if worth_staying or not opts else [])


def agent_decide(gs, player, can_move=True):
    """An agent picks one action and a line, steered by what's discovered and whether it leads the march."""
    h = heading(gs)
    here = [p["name"] for p in gs.alive() if p is not player] + [n.capitalize() for n in npcs_at(gs.location)]
    present = ", ".join(here) or "no one else"
    base = (f"You are {player['name']}, {player['class_desc']}, who is {player['personality']}, adventuring with "
            f"{roster(gs, player)}. Present with you right now: {present}. React to what just happened, then issue "
            "exactly ONE action (no lists, semicolons, or 'then'). Address only someone present by name; if you are "
            "alone, do not invent a team or address absent allies or an NPC who is not in this room. You do not know "
            "the way ahead in advance: learn what a place or prize demands by asking those who know or by trying and "
            "being turned back, then go fetch what is needed. A potion is your only heal in a fight and the road has "
            "enemies, so if one is here and the party carries none, take it. Never re-ask something already "
            "answered. Say one short in-character line, and never name your bearings, a compass, an objective, or "
            "any game term aloud. ")
    if can_move:
        rule = ("Let 'bearings' steer you: if its 'go_to' is set, travel there and nowhere else. If its 'known' is "
                "false you do not yet know the way to your current aim, so ask someone here who might know it; but if "
                "no one here can point you to THIS aim, or they have already told you all they know, wander an "
                "unvisited path and discover it yourself rather than lingering.")
    else:
        rule = ("The party travels together and you are not leading the march this turn, so do NOT travel. Do "
                "something useful where you stand: take a potion or other useful item that is here, talk to a "
                "companion or an NPC, or hold position.")
    usr = json.dumps({"now": world_context(gs, gs.location), "goals": known_goals(gs),
                      "bearings": h, "carrying_potion": "potion" in gs.inventory, "recent": gs.recent_memory()})
    out = config.work_struct(AgentTurn, [("system", base + rule), ("human", usr)], temperature=0.85)
    return (out.get("say", "") or "").strip(), (out.get("action", "look") or "look").strip()


def resolve_proposals(proposals, compass_go_to=None, compass_weight=3):
    """Pick the party destination by assertiveness-weighted vote; the compass gets a baseline voice."""
    if not proposals:
        return "stay"
    tally = {}
    if compass_go_to:
        tally[compass_go_to] = compass_weight  # the proven-correct route is not just a tie-break
    for _, dest, _, w in proposals:
        tally[dest] = tally.get(dest, 0) + w
    top = max(tally.values())
    leaders = [d for d, t in tally.items() if t == top]
    if len(leaders) == 1:
        return leaders[0]
    if compass_go_to in leaders:
        return compass_go_to
    return sorted(leaders)[0]


def negotiate_move(gs):
    """No-human navigation: each agent argues for a destination, the engine tallies. Yields Argument, returns (dest, proposals)."""
    opts = navigation_options(gs)
    h = heading(gs)
    proposals = []
    for p in [a for a in gs.alive() if a["is_agent"]]:
        sysm = (f"You are {p['name']}, a {p['caution']} {p['combat_focus']} adventurer. The party decides together "
                "where to go next. The 'bearings' field's 'go_to' is the way you already know leads toward the goal, "
                "so propose that UNLESS your nature gives a concrete reason to deviate: the cautious may want to fall "
                "back for light or healing before danger, the bold may want to press on. If 'go_to' is null you do "
                "not yet know the way: head into one of the unexplored rooms in 'bearings' to discover it, and do "
                "not loiter to re-ask someone who has no more to tell. 'stay' is only an option when it is in the "
                "options list. Do not propose a room because you guess it connects somewhere, trust what you already "
                "know for geography. Your reason is spoken aloud to your companions, so keep it in character and "
                "never name your bearings, a compass, or any game term. Propose ONE option with one short reason.")
        usr = json.dumps({"options": opts, "bearings": h, "goals": known_goals(gs),
                          "here": world_context(gs, gs.location), "recent": gs.recent_memory()})
        prop = config.work_struct(Proposal, [("system", sysm), ("human", usr)])
        dest = (prop.get("destination") or "").lower().strip()
        if dest not in opts:  # invalid pick falls back to the known route, else the first way out
            dest = h.get("go_to") if h.get("go_to") in opts else opts[0]
        w = clamp(p.get("assertiveness", 3), 1, 5, 3)
        reason = (prop.get("reason") or "").strip()
        proposals.append((p["name"], dest, reason, w))
        gs.remember(f"{p['name']} argues to head to {dest}")
        yield Argument(p["name"], dest, reason)
    return resolve_proposals(proposals, h.get("go_to")), proposals
