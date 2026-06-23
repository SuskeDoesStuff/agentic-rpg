"""The mutable per-session game state.

Everything that changes during a playthrough lives on a single ``GameState``
instance, never in module globals. One session owns one ``GameState``, so a UI
can hold an independent game per user just by keeping its own instance. The world
graph, gates, spells, and model config are immutable constants and live in their
own modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameState:
    """All mutable state for one playthrough."""

    location: str = "village"
    prev_location: str | None = None
    inventory: list = field(default_factory=list)
    quests: dict = field(default_factory=lambda: {"retrieve_amulet": "open", "slay_guardian": "open"})
    defeated: set = field(default_factory=set)
    visited: set = field(default_factory=set)
    facts: set = field(default_factory=set)  # objective locations the party has learned, by name
    flee_counts: dict = field(default_factory=dict)

    party: list = field(default_factory=list)
    memory: list = field(default_factory=list)
    turn: int = 0
    round_moved: bool = False

    # presentation toggles (dialogue/banter); drivers may flip these
    say: bool = True
    banter: bool = True

    # optional deterministic input queues, used by tests and scripted demos;
    # when non-empty they pre-empt live human input
    scripted_actions: list = field(default_factory=list)
    scripted_battle: list = field(default_factory=list)

    def remember(self, line: str, keep: int = 14) -> None:
        self.memory.append(line)
        del self.memory[:-keep]

    def recent_memory(self, n: int = 8) -> list:
        return self.memory[-n:]

    def alive(self) -> list:
        return [p for p in self.party if p["hp"] > 0]

    def quests_done(self) -> bool:
        return all(v == "done" for v in self.quests.values())
