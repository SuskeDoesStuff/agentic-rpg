"""The speech channel.

Speech is its own channel and never touches the world pipeline. A line is routed
by name to the addressed NPC or companion, or to the whole party (silent-skip)
when unaddressed. NPC replies are grounded only in facts derived from world state.
The leaf functions return text; the routing generators yield Dialogue events.
"""
from __future__ import annotations

import json
import re

from . import config
from .agents import open_goals
from .events import Dialogue
from .pipeline import PARSE_SYS
from .schemas import Intent
from .world import WORLD, items_in_room, npcs_at

MOVE_WORDS = {"go", "move", "head", "walk", "enter", "travel", "return", "back"}


def looks_like_move(text):
    """Cheap heuristic (no model call) to spot a travel command for the move-throttle re-prompt."""
    w = re.findall(r"[a-z]+", text.lower())
    return bool(w) and (w[0] in MOVE_WORDS or any(r in w for r in WORLD["rooms"]))


def detect_addressee(gs, message, speaker=""):
    """The name a line is aimed at, if any companion or present NPC is named in it."""
    words = set(re.findall(r"[a-z]+", message.lower()))
    names = [p["name"].lower() for p in gs.party if p["name"].lower() != speaker.lower()] + npcs_at(gs.location)
    return next((n for n in names if n in words), "")


def agent_say(gs, player, situation, allow_silence=True):
    """One short in-character line from an agent, addressed only to whoever is actually present."""
    here = [p["name"] for p in gs.alive() if p is not player] + [n.capitalize() for n in npcs_at(gs.location)]
    present = ", ".join(here) or "no one else"
    rule = "If you have nothing worth adding, reply with an empty line." if allow_silence else ""
    sysm = (f"You are {player['name']}, {player['class_desc']}, who is {player['personality']}. Present with you "
            f"right now: {present}. React with ONE short in-character line. Address only someone present by name; if "
            "you are alone, speak your own resolve and address no one. Never address an absent ally, a 'team' or "
            f"'party' that is not here, or an NPC who is not in this room. {rule} No quotes.")
    usr = json.dumps({"situation": situation, "recent": gs.recent_memory()})
    return config.work_text([("system", sysm), ("human", usr)], max_tokens=40, temperature=0.9).strip().strip('"')


def npc_reply(gs, npc, message, speaker):
    """An NPC's reply, grounded only in facts derived from world state; a quests NPC grants intel."""
    info = WORLD.get("npcs", {}).get(npc, {})
    role = info.get("role", f"a {npc}")
    facts = {}
    if info.get("knows") == "quests":
        facts["open_goals"] = open_goals(gs)
        gs.intel = True
    elif info.get("knows") == "wares":
        facts["wares_here"] = items_in_room(gs, gs.location)
        facts["note"] = "there is no coin here; travelers simply take what they need"
    sysm = (f"You are {npc}, {role}. A traveler addresses you. Reply in character in 1-2 sentences, somewhere "
            "between chatty and terse. 'facts' are your ONLY source of world knowledge; never invent places, "
            "items, people, or lore. 'recent' is just the latest talk: do not repeat advice you already gave word "
            "for word, you may note you have said it before.")
    usr = json.dumps({"traveler": speaker, "said": message, "facts": facts, "recent": gs.recent_memory(6)})
    return config.work_text([("system", sysm), ("human", usr)], max_tokens=70, temperature=0.7)


def parse_player(gs, text):
    """Classify a human's line; a 'say ' prefix or surrounding quotes route it to speech without a model call."""
    low = text.lower()
    if low.startswith("say "):
        msg = text[4:].strip()
        return {"action": "say", "target": detect_addressee(gs, msg), "message": msg}
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        msg = text[1:-1].strip()
        return {"action": "say", "target": detect_addressee(gs, msg), "message": msg}
    return config.work_struct(Intent, [("system", PARSE_SYS), ("human", text)])


def banter(gs, situation, exclude=None):
    """Each agent may react to a situation; yields a Dialogue per non-empty line."""
    if not gs.banter:
        return
    for p in gs.alive():
        if p is exclude or not p["is_agent"]:
            continue
        line = agent_say(gs, p, situation, allow_silence=True)
        if line:
            gs.remember(f'{p["name"]}: "{line}"')
            yield Dialogue(p["name"], line)


def handle_speech(gs, speaker, message, addressee="", announce=True):
    """Route a spoken line by name to an NPC or companion, else to the whole party. Yields Dialogue events."""
    addressee = (addressee or "").lower().strip()
    if not addressee:
        addressee = detect_addressee(gs, message, speaker)
    if announce:
        gs.remember(f'{speaker}: "{message}"')
        yield Dialogue(speaker, message)
    if addressee in npcs_at(gs.location):  # name -> that NPC speaks
        line = npc_reply(gs, addressee, message, speaker)
        gs.remember(f'{addressee}: "{line}"')
        yield Dialogue(addressee.capitalize(), line)
        return
    agent = next((p for p in gs.alive() if p["is_agent"] and p["name"].lower() == addressee), None)
    if agent:  # name -> that agent answers
        reply = agent_say(gs, agent, f'{speaker} says to you: "{message}"', allow_silence=False)
        if reply:
            gs.remember(f'{agent["name"]}: "{reply}"')
            yield Dialogue(agent["name"], reply)
        return
    for p in gs.alive():  # unaddressed -> whole party, silent-skip
        if not p["is_agent"] or p["name"].lower() == speaker.lower():
            continue
        reply = agent_say(gs, p, f'{speaker} said: "{message}"', allow_silence=True)
        if reply:
            gs.remember(f'{p["name"]}: "{reply}"')
            yield Dialogue(p["name"], reply)
