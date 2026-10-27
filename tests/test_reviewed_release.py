"""Release failures specified before implementing the Opus-reviewed release builder."""

import json

import pytest

from finetuning_labs.rewriter.evaluation import digest
from finetuning_labs.rewriter.reviewed_release import load_reviews, select_reviewed


def pair():
    row = {
        "input": "can we move the call to four",
        "output": "Can we move the call to four?",
        "mode": "/slack",
        "checks": "Preserve the question.",
    }
    meta = {
        "id": "new-one",
        "split": "train",
        "source_group": "new-one",
        "row_sha256": digest(row),
        "mechanical_issues": [],
        "context": {},
        "language": "en",
        "length": "short",
    }
    review = {
        "id": "new-one",
        "row_sha256": digest(row),
        "input_verdict": "pass",
        "output_verdict": "pass",
        "pair_verdict": "pass",
        "verdict": "pass",
        "critique": "The question and proposed time are preserved.",
    }
    return row, meta, review


def test_no_unreviewed_or_stale_row_can_be_released():
    row, meta, review = pair()
    assert select_reviewed([(row, meta)], {})[0] == []
    selected, excluded = select_reviewed([(row, meta)], {(review["id"], "wrong"): review})
    assert selected == [] and excluded
    selected, excluded = select_reviewed(
        [(row, meta)], {(review["id"], review["row_sha256"]): review}
    )
    assert len(selected) == 1 and not excluded


def test_review_cannot_hide_failing_dimension(tmp_path):
    _row, _meta, review = pair()
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"reviews": [review | {"pair_verdict": "fail"}]}))
    with pytest.raises(ValueError, match="contradicts"):
        load_reviews([path])


def test_duplicate_reviews_cannot_silently_conflict(tmp_path):
    _row, _meta, review = pair()
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"reviews": [review, review | {"critique": "Different finding."}]}))
    with pytest.raises(ValueError, match="conflicting"):
        load_reviews([path])


def test_mechanical_failure_cannot_be_rescued_by_opus_pass():
    row, meta, review = pair()
    meta["mechanical_issues"] = ["heldout_overlap"]
    selected, excluded = select_reviewed(
        [(row, meta)], {(review["id"], review["row_sha256"]): review}
    )
    assert not selected and excluded


def test_metadata_identity_must_match_the_exported_row():
    row, meta, _review = pair()
    with pytest.raises(ValueError, match="row identity"):
        select_reviewed([(row | {"output": "The call has moved."}, meta)], {})
