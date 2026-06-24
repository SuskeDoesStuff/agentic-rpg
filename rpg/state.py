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
    quests: dict = field(default_factory=dict)  # quest_id -> "active"/"done"; empty until acquired by talking
    defeated: set = field(default_factory=set)
    visited: set = field(default_factory=set)
    facts: set = field(default_factory=set)  # objective locations the party has learned, by name
    met: set = field(default_factory=set)    # rooms whose NPCs the party has already greeted
    flee_counts: dict = field(default_factory=dict)

    party: list = field(default_factory=list)
    memory: list = field(default_factory=list)      # events: narration, combat, quest markers, votes
    dialogue: list = field(default_factory=list)     # spoken lines only, in their own lane events cannot evict
    _seq: int = 0                                     # monotonic tick so the two lanes can be merged in true order
    turn: int = 0
    round_moved: bool = False

    # presentation toggles (dialogue/banter); drivers may flip these
    say: bool = True
    banter: bool = True

    # optional deterministic input queues, used by tests and scripted demos;
    # when non-empty they pre-empt live human input
    scripted_actions: list = field(default_factory=list)
    scripted_battle: list = field(default_factory=list)

    def remember(self, line: str, keep: int = 16) -> None:
        """Record a world event (narration, combat, a quest marker). Crowded lane; evicts fast."""
        self._seq += 1
        self.memory.append((self._seq, line))
        del self.memory[:-keep]

    def remember_speech(self, line: str, keep: int = 16) -> None:
        """Record a spoken line. Its own lane, so narration and combat can never push conversation out."""
        self._seq += 1
        self.dialogue.append((self._seq, line))
        del self.dialogue[:-keep]

    def recent_memory(self, n: int = 8, dialogue: int = 4) -> list:
        """The situational view a decision-maker sees: recent events plus a little recent talk, in true order."""
        merged = self.memory[-n:] + self.dialogue[-dialogue:]
        return [line for _, line in sorted(merged, key=lambda x: x[0])]

    def recent_dialogue(self, n: int = 8) -> list:
        """The conversation thread a speaker sees: only recent spoken lines, undiluted by narration."""
        return [line for _, line in self.dialogue[-n:]]

    def alive(self) -> list:
        return [p for p in self.party if p["hp"] > 0]
