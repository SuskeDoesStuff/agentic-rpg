"""Model configuration and the three chokepoints the whole engine talks through.

Everything that calls a model goes through ``work_struct`` (typed JSON on the mini
workhorse), ``judge_struct`` (typed JSON on the full judge), or ``work_text``
(free prose). The langchain client is created lazily, so importing the package or
running the test suite never needs langchain or an API key; tests monkeypatch
these three functions. Live play reads ``OPENAI_API_KEY`` at call time, and when
it is absent the helpers raise :class:`EngineOffline` so a UI can show a dormant
"demo offline" state instead of crashing. The provider is a single swap point
here, so a deployed demo can point at a different model without touching the
engine.
"""
from __future__ import annotations

import os
from functools import lru_cache

from . import tracing

WORK_MODEL = os.environ.get("RPG_WORK_MODEL", "gpt-5.4-mini")
JUDGE_MODEL = os.environ.get("RPG_JUDGE_MODEL", "gpt-5.4")


class EngineOffline(RuntimeError):
    """Raised when a model call is attempted with no API key configured."""


def has_key() -> bool:
    """Whether a key is configured; the UI uses this to render online vs offline."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@lru_cache(maxsize=2)
def _llm(model: str):
    if not has_key():
        raise EngineOffline("no OPENAI_API_KEY set; the demo is offline")
    from langchain_openai import ChatOpenAI  # imported lazily so the package imports without langchain

    return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"])


def work_struct(schema, messages, temperature=None):
    """Typed JSON from the workhorse model, with a temperature-free retry for reasoning models."""
    llm = _llm(WORK_MODEL)
    client = llm.bind(temperature=temperature) if temperature is not None else llm
    try:
        out = client.with_structured_output(schema).invoke(messages).model_dump()
    except Exception:
        out = _llm(WORK_MODEL).with_structured_output(schema).invoke(messages).model_dump()
    tracing.record(f"work_struct:{schema.__name__}", WORK_MODEL, messages, out)
    return out


def judge_struct(schema, messages):
    """Typed JSON from the full-tier judge model."""
    out = _llm(JUDGE_MODEL).with_structured_output(schema).invoke(messages).model_dump()
    tracing.record(f"judge_struct:{schema.__name__}", JUDGE_MODEL, messages, out)
    return out


def work_text(messages, max_tokens=160, temperature=None, label="work_text"):
    """Free prose from the workhorse model, degrading gracefully if binds are unsupported."""
    binds = {"max_tokens": max_tokens}
    if temperature is not None:
        binds["temperature"] = temperature
    try:
        out = _llm(WORK_MODEL).bind(**binds).invoke(messages).content
    except Exception:
        try:
            out = _llm(WORK_MODEL).bind(max_tokens=max_tokens).invoke(messages).content
        except Exception:
            out = _llm(WORK_MODEL).invoke(messages).content
    tracing.record(label, WORK_MODEL, messages, out)
    return out
