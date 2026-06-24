"""Memory has two lanes so dialogue stays legible.

Spoken lines live in their own lane that narration, combat, and quest markers
cannot evict; decision-makers still get a composed, time-ordered view of both.
These tests pin that separation, the eviction protection (the property that fixes
the goldfish dialogue), and that the composed view preserves true order.
"""
from __future__ import annotations

from rpg.state import GameState


def test_speech_and_events_live_in_separate_lanes():
    gs = GameState()
    gs.remember("(narration) the door creaks open")
    gs.remember_speech('Borin: "ready"')
    assert gs.recent_dialogue() == ['Borin: "ready"']              # only spoken lines
    composed = gs.recent_memory()
    assert "(narration) the door creaks open" in composed          # events in the composed view
    assert 'Borin: "ready"' in composed                            # plus a little recent talk


def test_events_never_evict_dialogue():
    gs = GameState()
    gs.remember_speech('Exys: "I saw something move in the trees"')
    for i in range(40):                                            # a flood well past the event cap
        gs.remember(f"(event {i})")
    assert 'Exys: "I saw something move in the trees"' in gs.recent_dialogue()  # the line survives


def test_recent_memory_preserves_true_order():
    gs = GameState()
    gs.remember("e1")
    gs.remember_speech("s1")
    gs.remember("e2")
    gs.remember_speech("s2")
    assert gs.recent_memory(n=8, dialogue=8) == ["e1", "s1", "e2", "s2"]  # interleaved by when they happened


def test_each_lane_caps_independently():
    gs = GameState()
    for i in range(30):
        gs.remember(f"e{i}")
        gs.remember_speech(f"s{i}")
    assert len(gs.memory) == 16 and len(gs.dialogue) == 16          # each lane keeps its own tail
    assert gs.recent_dialogue(100)[-1] == "s29"                     # newest spoken line retained
