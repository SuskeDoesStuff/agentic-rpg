"""Shared test fixtures.

The whole point of the three chokepoints is that the engine can be exercised with
no model and no API key. ``stub_models`` (autouse) replaces them with deterministic
functions, so every test below is pure, fast, and reproducible. ``run_gen`` drives a
generator to completion when no human input is expected.
"""
from __future__ import annotations

import json

import pytest

from rpg import config, players

ROOMS = ["village", "market", "tavern", "forest", "cave", "grove", "bridge", "ruins", "crypt", "chapel", "sanctum"]
ITEMS = ["torch", "potion", "key", "amulet", "ward"]
NPCS = ["elder", "merchant", "barkeep", "priest"]


def _intent(text):
    t = text.lower()
    if "take" in t or "grab" in t or "buy" in t:
        return {"action": "take", "target": next((i for i in ITEMS if i in t), ""), "message": ""}
    if any(w in t for w in ["go", "head", "enter", "cross", "move", "back", "walk", "travel"]):
        room = next((r for r in ROOMS if r in t), "")
        if not room and any(w in t for w in ["back", "return", "previous"]):
            room = "back"  # let the pipeline resolve this to the previous room
        return {"action": "move", "target": room, "message": ""}
    if "talk" in t or "ask" in t:
        return {"action": "talk", "target": next((n for n in NPCS if n in t), ""), "message": ""}
    return {"action": "look", "target": "", "message": ""}


def _resolution(usr):
    d = json.loads(usr)
    action = (d["action"].get("action") or "").lower()
    target = (d["action"].get("target") or "").lower()
    ctx = d["context"]
    out = {"move_to": None, "grant_item": None, "note": "ok"}
    if action == "move" and target in ctx["exits"]:
        out["move_to"] = target
    if action == "take" and target in ctx["items_here"]:
        out["grant_item"] = target
    return out


def _work_struct(schema, messages, temperature=None):
    name = schema.__name__
    usr = messages[-1][1]
    if name == "Intent":
        return _intent(usr)
    if name == "Resolution":
        return _resolution(usr)
    if name == "ClassStats":
        return {"name": "fighter", "max_hp": 22, "attack": 8, "max_mana": 0,
                "combat_focus": "balanced", "caution": "steady", "assertiveness": 3}
    if name == "AgentTurn":
        return {"say": "", "action": "look"}
    if name == "CombatMove":
        return {"move": "attack", "spell": ""}
    if name == "Proposal":
        return {"destination": "stay", "reason": "hold here"}
    if name == "Verdict":
        return {"score": 5, "violations": [], "reason": "clean"}
    return {}


def _work_text(messages, max_tokens=160, temperature=None, label="work_text"):
    return "The party presses onward through the gloom."


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    monkeypatch.setattr(config, "work_struct", _work_struct)
    monkeypatch.setattr(config, "judge_struct", lambda schema, messages: _work_struct(schema, messages))
    monkeypatch.setattr(config, "work_text", _work_text)


@pytest.fixture
def game():
    """A default solo-agent game state."""
    return players.new_game([
        players.make_player("Borin", "a sellsword", "gruff", is_agent=True,
                            stats={"name": "sellsword", "max_hp": 30, "attack": 8})
    ])


def run_gen(gen):
    """Drive a generator that needs no human input; return its return value."""
    to_send = None
    while True:
        try:
            gen.send(to_send)
        except StopIteration as e:
            return e.value
        to_send = None
