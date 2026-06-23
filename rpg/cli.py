"""A terminal driver for the engine.

It iterates the ``play`` generator, rendering display events and answering input
requests from stdin. The engine holds all logic; this file is pure presentation,
which is exactly what a web UI replaces without touching the core.
"""
from __future__ import annotations

from . import config, events
from .engine import play
from .players import disposition_line, make_player, new_game


def render(ev):
    if isinstance(ev, events.Narration):
        print(ev.text)
    elif isinstance(ev, events.Dialogue):
        print(f'  {ev.speaker}: "{ev.text}"')
    elif isinstance(ev, events.System):
        print(f"  {ev.text}")
    elif isinstance(ev, events.Argument):
        print(f'  {ev.speaker} argues for {ev.destination}: "{ev.reason}"')
    elif isinstance(ev, events.QuestUpdate):
        print(f"Quest: {ev.title} {ev.status}")
    elif isinstance(ev, events.GameOver):
        print(f"\n== {'all quests complete' if ev.won else 'game over'} ==")


def drive(gs, get_action=input, get_battle=None):
    """Run the engine to completion, sourcing human input from the given callables."""
    get_battle = get_battle or get_action
    gen = play(gs)
    to_send = None
    while True:
        try:
            ev = gen.send(to_send)
        except StopIteration:
            break
        to_send = None
        if isinstance(ev, events.NeedAction):
            if ev.note:
                print(f"  ({ev.note})")
            to_send = get_action(f"{ev.actor} > ")
        elif isinstance(ev, events.NeedBattleChoice):
            to_send = get_battle(f"  {ev.actor} - {', '.join(ev.options)}? ")
        else:
            render(ev)


def setup_wizard():
    """Build a party from typed character descriptions and return a fresh GameState."""
    def ask_int(prompt, lo, hi):
        while True:
            try:
                v = int(input(prompt))
                if lo <= v <= hi:
                    return v
            except ValueError:
                pass
            print(f"  enter a number {lo}-{hi}")

    n = ask_int("How many players (1-4)? ", 1, 4)
    agents = ask_int(f"How many of those are AI agents (0-{n})? ", 0, n)
    party = []
    for i in range(agents):
        print(f"-- agent {i + 1} --")
        name = input("  name: ").strip() or f"Agent {i + 1}"
        cls = input("  class description (free text): ").strip() or "a wandering adventurer"
        per = input("  personality: ").strip() or "steady"
        print("  rolling stats...")
        party.append(make_player(name, cls, per, is_agent=True))
    for i in range(n - agents):
        print(f"-- human {i + 1} --")
        name = input("  name: ").strip() or f"Player {i + 1}"
        cls = input("  class description (free text): ").strip() or "a wandering adventurer"
        per = input("  personality: ").strip() or "steady"
        print("  rolling stats...")
        party.append(make_player(name, cls, per, is_agent=False))
    gs = new_game(party)
    print("\nParty:", [(p["name"], p["class_name"], p["max_hp"], p["attack"],
                        "agent" if p["is_agent"] else "human") for p in gs.party])
    for p in gs.party:
        print("  " + disposition_line(p))
    return gs


def main():
    if not config.has_key():
        print("This demo is offline. Set OPENAI_API_KEY to play.")
        return
    drive(setup_wizard())


if __name__ == "__main__":
    main()
