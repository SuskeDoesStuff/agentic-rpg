"""A graph-grounded, multi-agent fantasy RPG game master.

Consistency lives in code (the world graph, deterministic combat and quests, the
validate and guardrail steps); the model handles language and a set of bounded,
consequential choices. The engine is driven as an event stream so the same core
plays in a terminal, in a web UI, or under a scripted test harness.
"""

from __future__ import annotations

from .config import EngineOffline, has_key
from .players import disposition_line, make_player, new_game
from .state import GameState
from .world import WORLD, build_world

__all__ = [
    "GameState",
    "WORLD",
    "build_world",
    "make_player",
    "new_game",
    "disposition_line",
    "has_key",
    "EngineOffline",
]
