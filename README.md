# Agentic RPG Game Master

A graph-grounded, multi-agent fantasy RPG played in text. One to four characters,
human or LLM agents, share a world, take turns, talk in character with each other
and with NPCs, complete interlocking quests, and fight a magic-and-mana battle
system where each fighter chooses its own moves.

The design principle runs through everything: **consistency lives in code, the
model handles language and bounded choices.** A NetworkX graph holds the world, a
LangGraph pipeline validates and guardrails every action, and combat and quests
resolve in pure Python. The model never decides who wins a fight, where a path
leads, or whether an item exists. It decides what to say, how to act within the
legal options, and, when there is no human in the party, where the group should go
next.

## Why this is agentic, and where the agency lives

Agents are not on rails. Each turn an agent reads the world state, its memory, the
party roster, and a graph-computed objective compass, then generates an action in
natural language. Its choices have consequences: an agent that fails to prepare or
fights past the point of winning can get the party killed, and a support-disposed
agent that heals the tank can carry a fight. The interesting decisions are
deliberately the model's (preparation, tactics, advocacy) while the scaffold holds
the line where correctness matters (geography, legality, numbers). That split is
the point, and it is the production-realistic pattern for LLM agents.

Highlights:

- **Dispositions from free text.** A class description and personality become
  clamped stats, mana, and a disposition (combat focus from the class, caution and
  council assertiveness from the personality) in one model call. How a character
  fights and argues falls out of how it was described.
- **Discovery-gated knowledge.** Agents only learn where things are after the party
  discovers them, by asking a knowledgeable NPC or by exploring, so NPCs are
  mechanically meaningful rather than flavor.
- **A real combat system.** Attack, cast (mana-costed spells), drink a potion,
  defend, or flee; HP and mana persist across fights; a deterministic outlook tells
  a fighter whether it is on track to win or to fall first, so survival can
  override temperament.
- **Multi-agent negotiation.** With no human present, agents argue for a
  destination and an assertiveness-weighted vote resolves it, with the compass as a
  baseline voice so a confident-but-wrong consensus needs real support to override
  the proven route.
- **One engine, many drivers.** The engine is a generator that emits display events
  and pauses for input, so a terminal, a web UI, and the test suite all drive the
  identical core with no duplicated logic.

## Architecture

```
rpg/
  state.py      GameState: all mutable per-session state, no globals
  config.py     model tiers and the 3 LLM chokepoints (lazy client, offline guard)
  schemas.py    Pydantic types for every structured model call
  world.py      the declarative world spec, the graph loader, graph helpers
  players.py    free-text -> clamped stats + mana + disposition
  pipeline.py   the per-turn LangGraph: parse, validate, resolve, guardrail, update, narrate
  speech.py     the speech channel: NPC and party dialogue, routed by name
  combat.py     spells, mana, potions, per-fighter choice, deterministic resolution
  agents.py     the compass, gated goals, decisions, and the negotiation
  engine.py     play(gs): the event-stream generator that drives a whole game
  cli.py        a terminal driver
```

Every model call goes through `config.work_struct`, `config.judge_struct`, or
`config.work_text`. The client is created lazily, so importing the package and
running the tests needs neither langchain nor an API key, and the test suite stubs
those three functions to drive the engine deterministically.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                 # deterministic suite, no key required
export OPENAI_API_KEY=...  # then play in the terminal
rpg
```

The model layer is swappable in `config.py`, and when no key is set the engine
reports as offline rather than crashing, so a hosted demo can be turned on and off
by adding or removing the key.

## Tests

The suite is the contract. It stubs the three model chokepoints and exercises the
world loader, the pipeline invariants and gates, combat and magic, the compass and
discovery gating, the negotiation resolver, the movement authority, and a full
scripted playthrough that must end with both quests complete and every invariant
intact. CI runs lint and the suite on every push.
