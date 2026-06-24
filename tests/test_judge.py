"""The evaluation harness is part of the contract.

The deterministic robustness suite is asserted as a property here, so weakening the
guardrail or the narration policer fails CI. The judge wiring is exercised through
the stubbed ``judge_struct`` (no key), and the live judge pass is confirmed to skip
cleanly when none is set.
"""
from __future__ import annotations

from rpg import config, evals, judge


def test_guardrail_detects_and_contains_every_fabrication():
    g = evals.guardrail_eval()
    assert g["injected"] > 0
    assert g["detected"] == g["injected"]    # every illegal move/grant is caught on the first pass
    assert g["contained"] == g["injected"]   # and reduced to a safe no-op once retries run out
    assert g["false_positives"] == 0         # legal outcomes are never blocked


def test_narration_policer_catches_leaks_and_passes_clean_lines():
    n = evals.narration_policer_eval()
    assert n["leaks"] > 0 and n["leaks_caught"] == n["leaks"]
    assert n["clean"] > 0 and n["clean_passed"] == n["clean"]


def test_endtoend_pipeline_blocks_a_cheating_model():
    e = evals.endtoend_guardrail_eval()
    assert e["turns"] > 0 and e["blocked"] == e["turns"]


def test_world_facts_group_real_entities():
    facts = judge.world_facts()
    assert "village" in facts["rooms"]
    assert "torch" in facts["items"]
    assert "elder" in facts["npcs"]
    assert "wolf" in facts["enemies"]


def test_grade_routes_through_judge_struct(game):
    v = judge.grade("The elder watches in silence.", kind="narration")
    assert "score" in v and "violations" in v  # shape comes back from the (stubbed) judge model


def test_summarize_aggregates_scores_and_violations():
    verdicts = [
        {"score": 10, "violations": []},
        {"score": 4, "violations": ["invented_entity", "leaked_mechanic"]},
        {"score": 7, "violations": ["invented_entity"]},
    ]
    s = judge.summarize(verdicts)
    assert s["n"] == 3
    assert s["mean_score"] == 7.0
    assert s["clean_rate"] == round(1 / 3, 3)
    assert s["violations"] == {"invented_entity": 2, "leaked_mechanic": 1}


def test_summarize_handles_empty():
    s = judge.summarize([])
    assert s["n"] == 0 and s["mean_score"] is None


def test_judge_eval_skips_without_a_key(monkeypatch):
    monkeypatch.setattr(config, "has_key", lambda: False)  # control the key, don't depend on the environment
    assert "skipped" in evals.judge_eval()


def test_judge_eval_runs_when_a_key_is_present(monkeypatch):
    monkeypatch.setattr(config, "has_key", lambda: True)   # the model calls are stubbed by conftest
    out = evals.judge_eval(max_rounds=2, cap=6)
    assert "skipped" not in out and "n" in out and out["clean_rate"] is not None
