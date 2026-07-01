"""The watch-only web driver, over the FastAPI test client.

The engine is exercised through the autouse model stubs, so these run without a
key. They pin the page, the world-graph endpoint, a full streamed game (which must
reach a terminal gameover and report at least one room), and the honest no-key
error path.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")  # the web extra is optional; skip rather than error when absent

from fastapi.testclient import TestClient  # noqa: E402

from rpg import config, web  # noqa: E402  (rpg.web imports fastapi, so it must follow the skip)
from rpg.world import WORLD  # noqa: E402

client = TestClient(web.app)


def _events(body: str):
    """Pull the JSON payloads out of an SSE response body."""
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: ") :]))
    return out


def test_index_serves_the_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Agentic RPG" in r.text  # the builder page, not an error
    assert "Watch LLM agents" in r.text  # the new tagline rendered


def test_map_returns_the_world_graph():
    r = client.get("/map")
    assert r.status_code == 200
    data = r.json()
    ids = {n["id"] for n in data["nodes"]}
    assert len(data["nodes"]) == len(WORLD["rooms"])  # every room is a node
    assert data["start"] in ids  # the start room is in the graph
    assert data["edges"]  # exits became drawable edges
    for n in data["nodes"]:  # coordinates normalized for the page
        assert 0.0 <= n["x"] <= 1.0 and 0.0 <= n["y"] <= 1.0


def test_play_streams_a_full_game(monkeypatch):
    monkeypatch.setattr(config, "has_key", lambda: True)  # pretend a key is present; stubs do the talking
    party = json.dumps(
        [
            {"name": "Borin", "class_desc": "a vanguard", "personality": "bold"},
            {"name": "Sable", "class_desc": "a battle-mage", "personality": "sharp"},
        ]
    )
    r = client.get("/play", params={"party": party, "rounds": 2})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    evs = _events(r.text)
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "start"  # opens with the party and start room
    assert "location" in kinds  # at least the starting room is reported
    assert "party" in kinds  # roster state is streamed
    assert kinds[-1] == "gameover"  # and it terminates cleanly
    start = next(e for e in evs if e["type"] == "start")
    assert len(start["party"]) == 2


def test_play_without_a_key_reports_honestly(monkeypatch):
    monkeypatch.setattr(config, "has_key", lambda: False)
    party = json.dumps([{"name": "Borin", "class_desc": "a vanguard", "personality": "bold"}])
    r = client.get("/play", params={"party": party, "rounds": 2})
    evs = _events(r.text)
    assert [e["type"] for e in evs] == ["error"]  # one honest error, no faked game
    assert "API key" in evs[0]["message"]


def test_play_surfaces_a_mid_game_failure(monkeypatch):
    monkeypatch.setattr(config, "has_key", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("model exploded")
        yield  # never reached; makes this a generator so it matches engine.play

    monkeypatch.setattr(web.engine, "play", boom)
    party = json.dumps([{"name": "Borin", "class_desc": "a vanguard", "personality": "bold"}])
    r = client.get("/play", params={"party": party, "rounds": 2})
    evs = _events(r.text)
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "start"  # party built, game opened
    assert kinds[-1] == "error"  # then the failure is shown, not swallowed
    assert "model exploded" in evs[-1]["message"]
