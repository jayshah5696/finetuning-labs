"""Reject weak reviews, missing decision dimensions, leakage and changed literals."""

import copy

import pytest

from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS
from finetuning_labs.rewriter.boost_data import (
    QUOTAS,
    SOL,
    boost_issues,
    focused_plans,
    review_packet,
    verified_pair,
    verify_holdouts,
    verify_saved_review,
)
from finetuning_labs.rewriter.evaluation import digest, saved_call
from finetuning_labs.rewriter.scale_data import VERIFY_SYSTEM
from finetuning_labs.rewriter.training import LeakageIndex


def pair():
    row = {
        "input": "um forward the sketch to Odel",
        "output": "Forward the sketch to Odel.",
        "mode": "/transcribe",
        "checks": "Keep the action.",
    }
    meta = {
        "id": "new-one",
        "split": "train",
        "task_family": "action_preservation",
        "language": "en",
        "context": {},
        "identity": False,
        "protected_literals": [],
        "native_accepted": True,
        "row_sha256": digest(row),
        "mechanical_issues": [],
        "decisions": {
            "labeling_engine." + k: {"score": 0.95, "confidence": 0.9} for k in DIMENSIONS
        },
    }
    review = {
        "id": meta["id"],
        "row_sha256": digest(row),
        "input_verdict": "pass",
        "output_verdict": "pass",
        "pair_verdict": "pass",
        "verdict": "pass",
        "critique": "The main action and recipient survive.",
    }
    return row, meta, review


def test_focused_source_plan_has_independent_training_groups_and_all_failure_targets():
    plans = focused_plans()
    assert sum(sum(v.values()) for v in QUOTAS.values()) == 500
    assert len({p["source_group"] for p in plans}) == len(plans)
    assert {p["split"] for p in plans} == {"train"}
    assert {p["task_family"] for p in plans} >= {
        "action_preservation",
        "spoken_correction",
        "conditions",
        "code_preservation",
        "literal_preservation",
        "long_completeness",
    }
    assert {p["length"] for p in plans} == {"short", "medium", "long"}


@pytest.mark.parametrize(
    "change", ["review_hash", "verdict", "missing_decision", "weak_decision", "mechanical"]
)
def test_only_exact_sol_passes_with_complete_strong_decisions_are_retained(change):
    row, meta, review = pair()
    assert verified_pair(row, meta, review)
    meta, review = copy.deepcopy(meta), copy.deepcopy(review)
    if change == "review_hash":
        review["row_sha256"] = "wrong"
    elif change == "verdict":
        review.update(pair_verdict="uncertain", verdict="uncertain")
    elif change == "missing_decision":
        meta["decisions"].pop("labeling_engine.pair_intent")
    elif change == "weak_decision":
        meta["decisions"]["labeling_engine.pair_facts"]["score"] = 0.5
    else:
        meta["mechanical_issues"] = ["heldout_overlap"]
    assert not verified_pair(row, meta, review)


def test_transcription_cannot_change_the_main_verb_or_drop_clean_words():
    row, meta, _ = pair()
    assert boost_issues(meta, row, LeakageIndex([])) == []
    row["output"] = "Read the sketch to Odel."
    assert "transcribe_lexical_change" in boost_issues(meta, row, LeakageIndex([]))


def test_native_uncertainty_requires_a_second_matching_sol_pass():
    row, meta, review = pair()
    meta["native_accepted"] = False
    meta["decisions"]["labeling_engine.pair_intent"].update(score=0.61, confidence=0.22)
    assert not verified_pair(row, meta, review)
    assert verified_pair(row, meta, review, second_review=review)
    assert not verified_pair(
        row, meta, review, second_review=review | {"pair_verdict": "fail", "verdict": "fail"}
    )


def test_confident_native_rejection_cannot_be_overridden_by_model_reviews():
    row, meta, review = pair()
    meta["decisions"]["labeling_engine.pair_facts"].update(score=0.01, confidence=0.98)
    assert not verified_pair(row, meta, review, second_review=review)


@pytest.mark.parametrize("changed", ["data/dev.jsonl", "sft/validation.jsonl"])
def test_in_place_update_cannot_change_frozen_holdout_bytes(changed):
    import hashlib

    files = {"data/dev.jsonl": b"dev", "sft/validation.jsonl": b"validation"}
    freeze = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    verify_holdouts(files, freeze)
    files[changed] += b"changed"
    with pytest.raises(ValueError, match="holdout"):
        verify_holdouts(files, freeze)


def test_new_examples_cannot_reuse_holdouts_or_change_protected_code():
    row, meta, _ = pair()
    assert "protected_overlap" in boost_issues(meta, row, LeakageIndex([row["input"]]))
    row = {
        "input": "explain this snippet\nlimit = 17",
        "output": "Explain this snippet.\nlimit = 18",
        "mode": "/prompt",
        "checks": "Keep code.",
    }
    meta = meta | {"task_family": "code_preservation", "protected_literals": ["limit = 17"]}
    assert "protected_literal_changed" in boost_issues(meta, row, LeakageIndex([]))


def test_release_requires_complete_quotas_and_unique_verified_sources():
    from finetuning_labs.rewriter.boost_data import validate_selection

    row, meta, review = pair()
    entry = {"row": row, "meta": meta, "review": review}
    with pytest.raises(ValueError, match="quotas"):
        validate_selection([entry])
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection([entry, entry])


def test_interrupted_directory_swap_restores_the_original_dataset(tmp_path, monkeypatch):
    import os

    from finetuning_labs.rewriter.boost_data import replace_dataset

    current, prepared = tmp_path / "current", tmp_path / "prepared"
    current.mkdir()
    prepared.mkdir()
    (current / "train").write_text("old")
    (prepared / "train").write_text("new")
    actual = os.replace

    def interrupt(source, destination):
        if str(source).endswith("-incoming-boost"):
            raise OSError("interrupted swap")
        return actual(source, destination)

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        replace_dataset(prepared, current)
    assert (current / "train").read_text() == "old"
    monkeypatch.setattr(os, "replace", actual)
    replace_dataset(prepared, current)
    assert (current / "train").read_text() == "new"


def test_topup_uses_new_source_ids_and_only_missing_quota_cells():
    from finetuning_labs.rewriter.boost_data import supplement_plans

    counts = {(m, length): n for m, cells in QUOTAS.items() for length, n in cells.items()}
    counts["/prompt", "long"] -= 2
    plans = supplement_plans(counts, 1)
    assert len(plans) == 12
    assert {(p["mode"], p["length"], p["split"]) for p in plans} == {("/prompt", "long", "train")}
    assert not {p["id"] for p in plans} & {p["id"] for p in focused_plans(3)}


def test_final_overlap_evidence_must_bind_to_the_published_rows():
    from finetuning_labs.rewriter.boost_data import verify_overlap_report

    row, _, _ = pair()
    report = {"rows": 1, "new_to_new_overlap": 0, "rows_sha256": digest([row])}
    verify_overlap_report([row], report)
    with pytest.raises(ValueError, match="overlap"):
        verify_overlap_report([row | {"output": "Changed."}], report)


@pytest.mark.parametrize("change", ["model", "packet", "verdict"])
def test_verification_must_come_from_the_exact_saved_sol_call(tmp_path, change):
    import json

    row, meta, review = pair()
    request = {
        "model": SOL,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": json.dumps([review_packet(row, meta)])},
        ],
    }
    response = {
        "model": SOL,
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps({"reviews": [review]})}}
        ],
    }
    record = saved_call(tmp_path / "review.json", request, lambda: response)
    assert verify_saved_review(row, meta, record) == review
    if change == "model":
        record["result"]["model"] = "other-model"
    elif change == "packet":
        record["request"]["messages"][1]["content"] = "[]"
        record["request_sha256"] = digest(record["request"])
    else:
        changed = review | {"pair_verdict": "fail", "verdict": "fail"}
        record["result"]["choices"][0]["message"]["content"] = json.dumps({"reviews": [changed]})
    record["result_sha256"] = digest(record["result"])
    if change == "verdict":
        assert verify_saved_review(row, meta, record)["verdict"] == "fail"
    else:
        with pytest.raises(ValueError, match="model|packet|request"):
            verify_saved_review(row, meta, record)
