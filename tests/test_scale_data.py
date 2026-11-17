"""Failure cases for scaling: repeated sources, held-out leakage, stale/partial reviews,
and release counts that hide a missing mode. Written before the scale implementation.
"""

import pytest

from finetuning_labs.rewriter.evaluation import digest
from finetuning_labs.rewriter.scale_data import expanded_plans, parse_reviews, select_balanced


def test_expansion_has_independent_briefs_and_frozen_splits():
    plans = expanded_plans()
    assert len(plans) == 1440
    assert len({p["id"] for p in plans}) == len(plans)
    assert len({p["brief"] for p in plans}) == len(plans)
    assert {p["length"] for p in plans if p["identity"]} == {"short", "medium", "long"}
    assert all(p["split"] in {"train", "dev"} for p in plans)
    assert len({p["domain"] for p in plans}) >= 60
    assert {p["language"] for p in plans} == {"en", "de", "pt"}


def packet():
    return [{"id": "one", "row_sha256": "a" * 64}]


def verdict(**changes):
    return {
        "id": "one",
        "row_sha256": "a" * 64,
        "input_verdict": "pass",
        "output_verdict": "pass",
        "pair_verdict": "pass",
        "verdict": "pass",
        "critique": "The requested wording is preserved.",
        **changes,
    }


@pytest.mark.parametrize(
    "reviews",
    [[], [verdict(), verdict()], [verdict(row_sha256="b" * 64)], [verdict(pair_verdict="fail")]],
)
def test_invalid_or_unbound_reviews_never_authorize_release(reviews):
    with pytest.raises(ValueError):
        parse_reviews(packet(), {"reviews": reviews})


def test_insufficient_mode_cannot_be_hidden_by_total_count():
    rows = [
        {"input": str(i), "output": str(i), "mode": "/slack", "checks": "Same."}
        for i in range(1000)
    ]
    meta = [
        {
            "id": str(i),
            "split": "train",
            "source_group": str(i),
            "row_sha256": digest(row),
            "mechanical_issues": [],
        }
        for i, row in enumerate(rows)
    ]
    with pytest.raises(ValueError, match="/transcribe"):
        select_balanced(list(zip(rows, meta, strict=True)), train_per_mode=250, dev_per_mode=25)


def test_review_candidates_requires_openrouter_api_key_env(tmp_path, monkeypatch):
    from finetuning_labs.rewriter.scale_data import review_candidates

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is required"):
        review_candidates(tmp_path)
