"""Typed schemas for every structured model call.

Using ``with_structured_output`` against these means parsing is types, not regex,
and a malformed model reply fails loudly at the boundary rather than corrupting
game state downstream.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Intent(BaseModel):
    action: str = Field(description="one of move, take, talk, look, use, say")
    target: str = Field(default="", description="noun for actions, or addressed name for say/talk")
    message: str = Field(default="", description="the spoken words when action is say or talk")


class Resolution(BaseModel):
    move_to: str | None = None
    grant_item: str | None = None
    note: str = ""


class ClassStats(BaseModel):
    name: str = Field(description="short class name")
    max_hp: int
    attack: int
    max_mana: int = Field(default=0, description="0 for pure martials, up to ~20 for dedicated casters")
    combat_focus: str = Field(default="balanced", description="offense, support, or balanced; from the class")
    caution: str = Field(default="steady", description="cautious, steady, or reckless; fragility plus personality")
    assertiveness: int = Field(default=3, description="1 defers, 5 dominates, in group decisions; from personality")


class AgentTurn(BaseModel):
    say: str = Field(description="one short in-character line")
    action: str = Field(description="a short action a player would type")


class CombatMove(BaseModel):
    move: str = Field(description="one of: attack, cast, potion, defend, flee")
    spell: str = Field(default="", description="the spell name when move is cast")


class Proposal(BaseModel):
    destination: str = Field(description="a room to travel to, or 'stay'")
    reason: str = Field(description="one short in-character clause")


class Verdict(BaseModel):
    score: int
    violations: list[str] = []
    reason: str = ""
