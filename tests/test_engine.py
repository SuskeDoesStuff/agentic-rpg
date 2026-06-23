"""The engine: movement authority, greeting on arrival, completion, and a full run."""
from __future__ import annotations

from rpg import engine, players, quests
from rpg.events import QuestUpdate
from tests.conftest import run_gen


def _agent(name, **stats):
    base = {"name": name.lower(), "max_hp": 24, "attack": 9}
    base.update(stats)
    return players.make_player(name, "an operative", "sharp", is_agent=True, stats=base)


def _human(name, **stats):
    base = {"name": name.lower(), "max_hp": 24, "attack": 9}
    base.update(stats)
    return players.make_player(name, "a warrior", "steady", is_agent=False, stats=base)


def _events(gen):
    """Drive a no-input generator and collect everything it yields."""
    out, to_send = [], None
    while True:
        try:
            out.append(gen.send(to_send))
        except StopIteration:
            return out
        to_send = None


def test_movement_authority_human_leads():
    gs = players.new_game([_human("Suske"), _agent("Exys")])
    assert engine.movement_allowed(gs, gs.party[0]) is True   # human may move
    assert engine.movement_allowed(gs, gs.party[1]) is False  # agent follows the human
    gs.round_moved = True
    assert engine.movement_allowed(gs, gs.party[0]) is False  # one move per round


def test_movement_authority_all_agents():
    gs = players.new_game([_agent("A"), _agent("B")])
    assert engine.movement_allowed(gs, gs.party[0]) is True   # first agent leads
    gs.round_moved = True
    assert engine.movement_allowed(gs, gs.party[1]) is False  # the rest hold


def test_local_work_gate():
    gs = players.new_game([_agent("A")])
    gs.location = "village"
    assert engine.room_has_local_work(gs) is True  # the elder is here
    gs.location = "bridge"
    gs.inventory = []
    assert engine.room_has_local_work(gs) is False  # empty corridor


def test_greeting_the_start_acquires_its_quests():
    gs = players.new_game([_agent("A")])
    gs.banter = False
    evs = _events(engine.greet_locals(gs))
    acquired = [e.title for e in evs if isinstance(e, QuestUpdate) and e.status == "acquired"]
    assert acquired == ["Recover the Amulet", "Kill the Guardian"]
    assert gs.quests == {"amulet_hunt": "active", "kill_guardian": "active"}
    assert "village" in gs.met
    assert _events(engine.greet_locals(gs)) == []  # greeting a met room is a no-op


# A full traversal: meet every giver as new rooms are found, fetch four items, kill three foes.
DEMO_ACTIONS = [
    "go tavern", "go village",                  # meet the barkeep -> the grove-bear bounty
    "go market", "take torch", "take potion", "go village",
    "go forest", "go grove", "go forest",       # the grove bear
    "go cave", "take key", "go forest", "go village",
    "go bridge", "go ruins", "take amulet",     # the amulet (the locked shrine needs the key)
    "go chapel", "take ward", "go ruins",        # meet the priest -> the sanctum rite, take the ward
    "go crypt", "go ruins",                      # the guardian (the dark crypt needs the torch)
    "go chapel", "go sanctum",                   # the wraith (the sealed sanctum needs the ward)
]


def test_full_scripted_run_completes_every_quest():
    gs = players.new_game([
        _agent("Borin", max_hp=36, attack=12, combat_focus="balanced"),
        _agent("Sable", max_hp=30, attack=12, max_mana=12, combat_focus="offense"),
    ])
    gs.say = False
    gs.banter = False
    run_gen(engine.greet_locals(gs))            # meet the village elder before setting out
    gs.scripted_actions = list(DEMO_ACTIONS)
    gs.scripted_battle = ["attack"] * 80
    while gs.scripted_actions:
        p = gs.party[gs.turn % len(gs.party)]
        run_gen(engine.take_turn(gs, p))
    assert gs.location == "sanctum"
    assert quests.all_done(gs)
    assert {"wolf", "bear", "guardian", "wraith"} <= gs.defeated
    assert all(0 <= p["hp"] <= p["max_hp"] and 0 <= p["mana"] <= p["max_mana"] for p in gs.party)


def test_second_flee_from_same_enemy_is_a_stalemate():
    gs = players.new_game([_agent("Solo", max_hp=18, attack=10)])
    gs.banter = False
    gs.inventory = ["torch"]  # lit, so entry is never the blocker here
    gs.scripted_battle = ["flee"]
    gs.location = "crypt"
    assert run_gen(engine.arrive(gs, "ruins", mover=None)) == "ok"  # first flee is allowed
    assert gs.flee_counts["guardian"] == 1
    gs.scripted_battle = ["flee"]
    gs.location = "crypt"
    assert run_gen(engine.arrive(gs, "ruins", mover=None)) == "stalemate"  # second flee ends it
    assert gs.flee_counts["guardian"] == 2
