"""Optional Langfuse tracing for the three model chokepoints.

Strictly opt-in and best-effort. With no Langfuse keys and no injected sink, every
function here is a no-op, the ``langfuse`` package is never imported, and the engine
behaves exactly as before. Any tracing error is swallowed, so observability can
never break gameplay. One game is one session (a root span); each model call is a
generation under it, named by kind and model.

The SDK is reached only through a small internal sink interface, a ``session()`` and
``flush()`` on the sink and ``record()`` / ``end()`` on a session, so all SDK
specifics live in one adapter (:class:`_LangfuseSink`). A version bump touches only
that adapter, and tests inject a fake sink of the same shape to exercise the enabled
path with no SDK and no keys.

Enable by setting ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` (and optionally
``LANGFUSE_HOST``) in the environment, or by calling :func:`configure` with a sink.
"""

from __future__ import annotations

import os
from contextvars import ContextVar

_sink = None  # injected sink; when set, tracing is on
_lazy = None  # cached real Langfuse-backed sink
_current: ContextVar = ContextVar("rpg_trace_session", default=None)


def configure(sink) -> None:
    """Inject a sink (web layer, tests). Pass ``None`` to disable and reset."""
    global _sink, _lazy
    _sink = sink
    _lazy = None


def enabled() -> bool:
    """Whether tracing should run: an injected sink, or Langfuse keys in the environment."""
    return _sink is not None or bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


def _get_sink():
    if _sink is not None:
        return _sink
    global _lazy
    if _lazy is None:
        from langfuse import Langfuse  # lazy: the package is needed only when tracing is enabled

        _lazy = _LangfuseSink(Langfuse())
    return _lazy


def begin_session(name: str, **meta):
    """Open a session for one game and make it current; returns an opaque token for :func:`end_session`."""
    if not enabled():
        return None
    try:
        session = _get_sink().session(name, meta or None)
    except Exception:
        return None
    return _current.set(session)


def end_session(token) -> None:
    """End the current session and flush pending events. Never raises."""
    if token is None:
        return
    session = _current.get()
    try:
        _current.reset(token)
    except Exception:
        pass
    try:
        if session is not None:
            session.end()
    except Exception:
        pass
    try:
        _get_sink().flush()
    except Exception:
        pass


def record(name: str, model: str, inp, output, **meta) -> None:
    """Record one model call as a generation under the current session, or standalone. Never raises."""
    if not enabled():
        return
    try:
        session = _current.get()
        target = session if session is not None else _get_sink()
        target.record(name, model, inp, output, meta or None)
    except Exception:
        pass  # observability must never break play


class _LangfuseSession:
    """A game's root span; each recorded call becomes a child generation under it."""

    def __init__(self, root):
        self._root = root

    def record(self, name, model, inp, output, metadata):
        self._root.start_observation(
            name=name, as_type="generation", input=inp, output=output, model=model, metadata=metadata
        ).end()

    def end(self):
        self._root.end()


class _LangfuseSink:
    """Adapter over the Langfuse v3+/v4 client; all SDK specifics are confined here."""

    def __init__(self, client):
        self._client = client

    def session(self, name, metadata):
        root = self._client.start_observation(name=name, as_type="span", metadata=metadata)
        return _LangfuseSession(root)

    def record(self, name, model, inp, output, metadata):  # a call made outside any session
        self._client.start_observation(
            name=name, as_type="generation", input=inp, output=output, model=model, metadata=metadata
        ).end()

    def flush(self):
        self._client.flush()
