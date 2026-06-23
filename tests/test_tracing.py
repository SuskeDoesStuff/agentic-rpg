"""Tracing is opt-in observability that must never change behaviour or crash play.

Two layers are tested. The sink-interface layer (a fake sink) covers the no-op,
record-under-session, standalone-record, error-swallowing, chokepoint, and engine
wiring. A separate adapter test drives ``tracing._LangfuseSink`` against a client
shaped like the real Langfuse v3+/v4 API (``start_observation`` with an ``as_type``,
child ``start_observation``, ``end``, ``flush``), so the SDK mapping is verified with
no real SDK and no keys.
"""
from __future__ import annotations

import pytest

from rpg import config, engine, players, tracing
from rpg.schemas import Intent

# Capture the real chokepoints at import, before the autouse model stub replaces them per test.
_REAL_WORK_STRUCT = config.work_struct
_REAL_WORK_TEXT = config.work_text


class FakeSession:
    def __init__(self, sink, name, metadata):
        self.sink, self.name, self.metadata, self.ended = sink, name, metadata, False

    def record(self, name, model, inp, output, metadata):
        self.sink.records.append({"trace": self.name, "name": name, "model": model,
                                  "input": inp, "output": output, "meta": metadata})

    def end(self):
        self.ended = True


class FakeSink:
    def __init__(self):
        self.records, self.sessions, self.flushed = [], [], 0

    def session(self, name, metadata):
        s = FakeSession(self, name, metadata)
        self.sessions.append(s)
        return s

    def record(self, name, model, inp, output, metadata):  # a call made outside any session
        self.records.append({"trace": None, "name": name, "model": model,
                             "input": inp, "output": output, "meta": metadata})

    def flush(self):
        self.flushed += 1


@pytest.fixture(autouse=True)
def _reset_tracing():
    tracing.configure(None)
    yield
    tracing.configure(None)


def test_disabled_is_a_strict_no_op():
    assert tracing.enabled() is False
    assert tracing.begin_session("x", a=1) is None
    tracing.record("k", "m", "in", "out")  # no sink: silent
    tracing.end_session(None)


def test_records_one_generation_per_call_under_a_session():
    sink = FakeSink()
    tracing.configure(sink)
    assert tracing.enabled() is True
    token = tracing.begin_session("rpg-game", party=["A", "B"])
    tracing.record("work_struct:Intent", "gpt", ["msgs"], {"action": "look"}, schema="Intent")
    tracing.end_session(token)
    assert sink.sessions and sink.sessions[0].name == "rpg-game"
    assert sink.sessions[0].ended is True and sink.flushed == 1
    assert len(sink.records) == 1
    r = sink.records[0]
    assert r["trace"] == "rpg-game" and r["name"] == "work_struct:Intent" and r["output"] == {"action": "look"}


def test_record_without_a_session_is_standalone():
    sink = FakeSink()
    tracing.configure(sink)
    tracing.record("work_text:setup", "gpt", ["m"], "prose")  # no begin_session
    assert len(sink.records) == 1 and sink.records[0]["trace"] is None


def test_backend_errors_never_propagate():
    class Boom:
        def session(self, *a, **k):
            raise RuntimeError("down")

        def record(self, *a, **k):
            raise RuntimeError("down")

        def flush(self):
            raise RuntimeError("down")

    tracing.configure(Boom())
    token = tracing.begin_session("g")     # swallowed -> None
    assert token is None
    tracing.record("k", "m", "i", "o")     # swallowed (standalone path raises, caught)
    tracing.end_session("not-a-token")     # swallowed


def test_real_chokepoints_emit_generations(monkeypatch):
    class _Struct:
        def invoke(self, messages):
            return type("R", (), {"model_dump": lambda self: {"action": "look", "target": "", "message": ""}})()

    class _LLM:
        def bind(self, **k):
            return self

        def with_structured_output(self, schema):
            return _Struct()

        def invoke(self, messages):
            return type("C", (), {"content": "a quiet line"})()

    monkeypatch.setattr(config, "_llm", lambda model: _LLM())
    sink = FakeSink()
    tracing.configure(sink)
    out = _REAL_WORK_STRUCT(Intent, [("human", "hi")])
    txt = _REAL_WORK_TEXT([("human", "hi")], label="narrate")
    assert out["action"] == "look" and txt == "a quiet line"
    names = [r["name"] for r in sink.records]
    assert "work_struct:Intent" in names and "narrate" in names


def test_play_opens_and_closes_a_session():
    sink = FakeSink()
    tracing.configure(sink)
    gs = players.new_game([players.make_player("A", "scout", is_agent=True,
                                               stats={"name": "s", "max_hp": 20, "attack": 8})])
    gs.banter = False
    gen = engine.play(gs, max_rounds=1)
    to_send = None
    while True:
        try:
            gen.send(to_send)
        except StopIteration:
            break
        to_send = None
    assert sink.sessions and sink.sessions[0].name == "rpg-game"
    assert sink.sessions[0].metadata["party"] == ["A"]
    assert sink.sessions[0].metadata["models"] == {"work": config.WORK_MODEL, "judge": config.JUDGE_MODEL}
    assert sink.sessions[0].ended is True and sink.flushed >= 1


# --- adapter: confirm the real Langfuse v3+/v4 surface is spoken correctly ---

class _V4Gen:
    def __init__(self, log, kw):
        self.log, self.kw = log, kw

    def end(self, **k):
        self.log.append(("gen_end", self.kw))


class _V4Span:
    def __init__(self, log, kw):
        self.log, self.kw = log, kw

    def start_observation(self, **kw):
        self.log.append(("child", kw))
        return _V4Gen(self.log, kw)

    def end(self, **k):
        self.log.append(("span_end", self.kw))


class _V4Client:
    def __init__(self):
        self.log, self.flushed = [], 0

    def start_observation(self, **kw):
        self.log.append(("root", kw))
        return _V4Span(self.log, kw)

    def flush(self):
        self.flushed += 1


def test_langfuse_adapter_speaks_the_v4_api():
    c = _V4Client()
    sink = tracing._LangfuseSink(c)
    session = sink.session("rpg-game", {"party": ["A"]})
    session.record("work_struct:Intent", "gpt", ["m"], {"action": "look"}, {"schema": "Intent"})
    session.end()
    sink.flush()
    shape = [(k, kw.get("as_type"), kw.get("name")) for k, kw in c.log if k in ("root", "child")]
    assert ("root", "span", "rpg-game") in shape
    assert ("child", "generation", "work_struct:Intent") in shape
    assert any(k == "gen_end" for k, _ in c.log) and any(k == "span_end" for k, _ in c.log)
    assert c.flushed == 1
