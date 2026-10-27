"""Failures specified before decision-validation implementation.

Do not send reference answers or fixture labels. Do not truncate long evidence. A missing,
uncertain, invalid, or failed decision cannot pass. One bad criterion cannot be averaged away.
Keep false acceptance visible and never silently repeat a paid call on resume.
"""

import json

import pytest

from finetuning_labs.rewriter.benchmark import Case
from finetuning_labs.rewriter.decision_validation import (
    CRITERIA,
    decide_one,
    decision_state,
    fixture_metrics,
    probability_verdict,
)


def sample():
    return Case(
        "one", "The count is not confirmed.", "HIDDEN GOLD ANSWER", "/slack", "HIDDEN EXPECTATION"
    )


def result(**changes):
    return {
        "model": "openai/gpt-6-luna-decisions-20261006",
        "answers": {name: {"type": "yes_no", "probability": 0.99} for name in CRITERIA},
        "requests_sent": 1,
        "cache_hits": 0,
        **changes,
    }


def test_remote_state_is_blind_and_keeps_context():
    state = decision_state(sample(), "The count is not confirmed.")
    assert "HIDDEN GOLD" not in state and "HIDDEN EXPECTATION" not in state
    payload = json.loads(state)
    assert set(payload) == {
        "input",
        "candidate",
        "mode",
        "mode_instructions",
        "context",
        "language",
    }
    assert payload["input"] == sample().input


def test_all_criteria_must_pass_with_uncertainty_preserved():
    values = dict.fromkeys(CRITERIA, 0.99)
    assert probability_verdict(values) == "pass"
    assert probability_verdict(values | {"meaning": 0.1}) == "fail"
    assert probability_verdict(values | {"meaning": 0.9}) == "uncertain"
    for wrong in [
        values | {"meaning": float("nan")},
        values | {"meaning": True},
        {},
        values | {"extra": 0.99},
    ]:
        with pytest.raises(ValueError):
            probability_verdict(wrong)


def test_saved_decision_resume_makes_no_extra_call(tmp_path):
    calls = []
    invoke = lambda state: calls.append(state) or result()
    config = {"model": "openai/gpt-6-luna-decisions", "max_state_chars": 100000}
    path = tmp_path / "call.json"
    first = decide_one(sample(), "Candidate.", path, config, invoke)
    assert first["status"] == "pass"
    assert decide_one(sample(), "Candidate.", path, config, invoke) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="differs"):
        decide_one(sample(), "Changed.", path, config, invoke)


def test_overflow_error_and_wrong_model_cannot_pass(tmp_path):
    config = {"model": "openai/gpt-6-luna-decisions", "max_state_chars": 2}

    def forbidden(state):
        raise AssertionError("Must not send truncated evidence")

    item = decide_one(sample(), "Candidate.", tmp_path / "long.json", config, forbidden)
    assert item["status"] == "context_overflow"
    config["max_state_chars"] = 100000
    item = decide_one(
        sample(),
        "Candidate.",
        tmp_path / "wrong.json",
        config,
        lambda state: result(model="some-other-model"),
    )
    assert item["status"] == "decision_error"
    item = decide_one(
        sample(), "Candidate.", tmp_path / "missing.json", config, lambda state: result(answers={})
    )
    assert item["status"] == "decision_error"


def test_false_acceptance_and_unresolved_controls_stay_visible():
    items = [
        {"expected": "pass", "status": "pass", "mechanical_failures": []},
        {"expected": "pass", "status": "uncertain", "mechanical_failures": []},
        {"expected": "fail", "status": "pass", "mechanical_failures": []},
        {"expected": "fail", "status": "fail", "mechanical_failures": ["literal"]},
        {"expected": "fail", "status": "decision_error", "mechanical_failures": []},
    ]
    report = fixture_metrics(items)
    assert report["scheduled"] == 5
    assert report["bad_false_accepts"] == 1
    assert report["bad_false_acceptance_rate"] == pytest.approx(1 / 3)
    assert report["semantic_only_bad_false_accepts"] == 1
    assert report["unresolved"] == 2
