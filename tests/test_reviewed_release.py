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


def test_export_release_uses_atomic_json_writes_and_deterministic_manifest(tmp_path, monkeypatch):
    import finetuning_labs.rewriter.reviewed_release as rr
    from finetuning_labs.rewriter.evaluation import write_json

    pool_root = tmp_path / "pool"
    pool_root.mkdir()
    rows, metas, reviews = [], [], []
    for split, text in [
        ("train", "Can we move the call to four?"),
        ("dev", "Is the staging server ready?"),
        ("validation", "Did the migration finish cleanly?"),
    ]:
        r = {
            "input": text.lower().rstrip("?"),
            "output": text,
            "mode": "/slack",
            "checks": "Preserve the question.",
        }
        sha = digest(r)
        rows.append(r)
        metas.append(
            {
                "id": f"case-{split}",
                "split": split,
                "source_group": f"group-{split}",
                "row_sha256": sha,
                "mechanical_issues": [],
                "context": {},
                "language": "en",
                "length": "short",
            }
        )
        reviews.append(
            {
                "id": f"case-{split}",
                "row_sha256": sha,
                "input_verdict": "pass",
                "output_verdict": "pass",
                "pair_verdict": "pass",
                "verdict": "pass",
                "critique": f"Question preserved for {split}.",
            }
        )
    (pool_root / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (pool_root / "scored-supporting.json").write_text(json.dumps(metas))
    rev_path = tmp_path / "reviews.json"
    rev_path.write_text(json.dumps({"reviews": reviews}))

    written_paths = []
    orig_write_json = write_json

    def spy_write_json(path, value):
        written_paths.append(path.name)
        return orig_write_json(path, value)

    monkeypatch.setattr(rr, "write_json", spy_write_json, raising=False)
    monkeypatch.setattr(rr, "_tokenizer", lambda: object())
    monkeypatch.setattr(
        rr,
        "token_audit",
        lambda ex, tok: {"input_tokens": 10, "chat_tokens": 30, "prompt_tokens": 20},
    )

    dest = tmp_path / "release"
    rr.export_release(pool_root, [rev_path], dest)
    assert {"supporting.json", "excluded.json", "quality.json", "manifest.json"} <= set(
        written_paths
    )
    manifest = json.loads((dest / "manifest.json").read_text())
    file_keys = list(manifest["files"].keys())
    assert file_keys == sorted(file_keys)
    assert "manifest.json" not in file_keys
