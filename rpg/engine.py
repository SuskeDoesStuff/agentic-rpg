"""The engine: one generator that drives a whole playthrough as an event stream.

``play(gs)`` yields display events and pauses on input requests, so a terminal, a
web UI, or a scripted test all drive the identical core. The party moves as one
body: with a human present the human leads and agents follow; with no human the
agents negotiate each move. Movement is one shared step per round.
"""

from __future__ import annotations

import json

from . import config, quests, tracing
from .agents import agent_decide, negotiate_move
from .combat import run_battle
from .events import Dialogue, GameOver, Narration, NeedAction, QuestUpdate, System
from .pipeline import run_turn
from .speech import agent_say, banter, detect_addressee, handle_speech, looks_like_move, npc_exchange, parse_player
from .world import enemy_in_room, items_in_room, npcs_at, world_context


def movement_allowed(gs, player):
    """Who may move the shared party this round: a human leads and agents follow, else the first agent leads."""
    if any(not q["is_agent"] for q in gs.party):
        return (not player["is_agent"]) and (not gs.round_moved)
    return not gs.round_moved


def room_has_local_work(gs):
    """Is there anything worth a local turn here: an ungrabbed item, or an NPC to ask?"""
    return bool(items_in_room(gs, gs.location) or npcs_at(gs.location))


def announce_completions(gs):
    """Emit a completion line for every quest whose objectives are now all secured."""
    for title in quests.complete(gs):
        gs.remember(f"(quest complete: {title})")
        yield QuestUpdate(title, "completed")


def greet_locals(gs):
    """On first arrival in a room, greet every NPC there: acquire its quests and learn its leads."""
    if gs.location in gs.met:
        return
    gs.met.add(gs.location)
    for npc in npcs_at(gs.location):
        yield from npc_exchange(gs, npc, "the party arrives and greets you", "the party")


def arrive(gs, prev, mover=None):
    """Shared post-move handling: mark moved, fight if guarded, then meet locals. Returns 'ok'/'gameover'/'stalemate'."""
    if gs.location == prev:
        return "ok"
    gs.round_moved = True
    yield from banter(gs, f"the party has entered the {gs.location}", exclude=mover)
    enemy = enemy_in_room(gs, gs.location)
    if enemy:
        outcome = yield from run_battle(gs, enemy, prev)
        if outcome == "lost":
            return "gameover"
        if outcome == "fled" and gs.flee_counts.get(enemy, 0) >= 2:
            return "stalemate"  # a second flee from the same foe means it cannot be beaten this way
        if outcome == "fled":
            return "ok"  # withdrew but may return; enemy rooms hold no NPCs to greet
    yield from announce_completions(gs)  # a kill here may finish a quest
    yield from greet_locals(gs)  # meet anyone new in this room
    return "ok"


def take_turn(gs, player):
    """One player's turn: speech to dialogue, or one world action, party moving as one. Returns 'ok' or 'gameover'."""
    if player["hp"] <= 0:
        return "ok"
    scripted = bool(gs.scripted_actions)
    can_move = True if scripted else movement_allowed(gs, player)  # a demo script is authoritative
    say = None
    preparsed = None
    if scripted:
        action = gs.scripted_actions.pop(0)
        if player["is_agent"] and gs.say:
            say = agent_say(gs, player, f"you are about to: {action}", allow_silence=False)
    elif player["is_agent"]:
        say, action = agent_decide(gs, player, can_move)
    else:
        note = ""
        while True:
            action = (yield NeedAction(player["name"], note)).strip()
            if ";" in action or " then " in action:
                note = "one action at a time, please"
                continue
            if (not can_move) and looks_like_move(action):
                note = "the party already moved together this round; act where you stand"
                continue
            break
    if say:
        gs.remember_speech(f'{player["name"]}: "{say}"')
        yield Dialogue(player["name"], say)

    if action:  # speech is its own channel and never hits the world pipeline
        intent = parse_player(gs, action)
        if intent["action"] in ("say", "talk"):
            agent_line = player["is_agent"] and bool(say)  # the agent already spoke its line above
            msg = say if agent_line else (intent.get("message") or action)
            yield from handle_speech(gs, player["name"], msg, intent.get("target", ""), announce=not agent_line)
            return "ok"
        if intent["action"] == "move" and not can_move:  # party moves together; a follower holds position
            yield System(f"{player['name']} stays with the party")
            return "ok"
        preparsed = intent  # reuse the classification for the world action

    if say and player["is_agent"]:  # a line aimed at a present NPC gets a full exchange, even mid-action
        npc = detect_addressee(gs, say, player["name"])
        if npc in npcs_at(gs.location):
            yield from npc_exchange(gs, npc, say, player["name"])

    prev = gs.location
    gs.turn += 1
    res = run_turn(gs, action, player["name"], intent=preparsed)
    yield Narration(res.get("narration", "..."))
    gs.remember(f"({res.get('narration', '')})")
    yield from announce_completions(gs)  # taking the objective item may finish a quest
    if (yield from arrive(gs, prev, mover=player)) == "gameover":
        return "gameover"
    if quests.all_done(gs):
        yield from banter(gs, "the party has completed every quest")
    return "ok"


def party_navigation_phase(gs):
    """No-human rounds: agents negotiate the party's move, then it executes once. Returns 'ok' or 'gameover'."""
    dest, proposals = yield from negotiate_move(gs)
    gs.round_moved = True  # the round's movement decision is made, move or stay
    if dest == "stay" or dest == gs.location:
        yield System("the party holds position.")
        return "ok"
    prev = gs.location
    lead = max(proposals, key=lambda x: x[3])[0] if proposals else "the party"
    gs.turn += 1
    res = run_turn(gs, f"go {dest}", lead)
    yield System(f"{lead} leads the way toward the {dest}")
    yield Narration(res.get("narration", "..."))
    gs.remember(f"({res.get('narration', '')})")
    return (yield from arrive(gs, prev, mover=None))


def narrate_opening(gs):
    """Two or three sentences setting the opening scene from world facts only."""
    party = ", ".join(f"{p['name']} ({p['class_name']})" for p in gs.party)
    sysm = (
        "You are a fantasy narrator. In 2-3 sentences set the opening scene: introduce the party and describe "
        "their starting room and the paths and people in it. Use only what is in the context, invent nothing."
    )
    text = config.work_text(
        [("system", sysm), ("human", json.dumps({"party": party, "room": world_context(gs, gs.location)}))],
        max_tokens=120,
        temperature=0.85,
        label="narrate_opening",
    )
    gs.remember(f"(opening) {text}")
    return text


def play(gs, max_rounds=48):
    """Drive a whole playthrough as a stream of events, under one tracing session. This is the engine."""
    token = tracing.begin_session(
        "rpg-game", party=[p["name"] for p in gs.party], models={"work": config.WORK_MODEL, "judge": config.JUDGE_MODEL}
    )
    try:
        yield from _play(gs, max_rounds)
    finally:
        tracing.end_session(token)


def _play(gs, max_rounds=48):
    yield Narration(narrate_opening(gs))
    yield from greet_locals(gs)  # meet the starting room's locals before the first round
    all_agent = not any(not p["is_agent"] for p in gs.party)
    for _ in range(max_rounds):
        gs.round_moved = False
        if all_agent:
            result = yield from party_navigation_phase(gs)
            if result == "gameover":
                yield GameOver(False, "the party has fallen")
                return
            if result == "stalemate":
                yield GameOver(False, "the party cannot overcome what blocks the way and falls back for good")
                return
            if quests.all_done(gs):
                yield GameOver(True, "all quests complete")
                return
            if not room_has_local_work(gs):  # empty room after travel: skip the hold-position chatter
                continue
        for p in gs.party:
            if not gs.alive():
                yield GameOver(False, "the party has fallen")
                return
            if (yield from take_turn(gs, p)) == "gameover":
                yield GameOver(False, "the party has fallen")
                return
        if quests.all_done(gs):
            yield GameOver(True, "all quests complete")
            return
    yield GameOver(quests.all_done(gs), "out of rounds")
