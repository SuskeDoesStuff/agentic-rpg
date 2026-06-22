"""The event vocabulary the engine emits.

The engine is a generator. It ``yield``s display events (narration, dialogue,
system lines, negotiation arguments, game-over) which a driver renders, and it
``yield``s input-request events (need-action, need-battle-choice) which a driver
answers by resuming the generator with ``.send(value)``. The same engine therefore
drives a terminal, a web UI, or a scripted test, with no duplicated logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Narration:
    """Scene-setting or action narration from the game master."""
    text: str


@dataclass
class Dialogue:
    """A line spoken by a player, agent, or NPC."""
    speaker: str
    text: str


@dataclass
class System:
    """A mechanical line: battle blows, a defeat, a follower holding position."""
    text: str


@dataclass
class Argument:
    """One agent's case during a no-human navigation vote."""
    speaker: str
    destination: str
    reason: str


@dataclass
class NeedAction:
    """Request a human's world action; resume the engine with the typed string."""
    actor: str
    note: str = ""


@dataclass
class NeedBattleChoice:
    """Request a human's battle move; resume with one of the option strings."""
    actor: str
    options: list = field(default_factory=list)


@dataclass
class GameOver:
    """Terminal event: the run has ended."""
    won: bool
    reason: str = ""


INPUT_EVENTS = (NeedAction, NeedBattleChoice)
