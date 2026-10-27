"""Publication failures specified before implementing the reviewed-data publisher.

Never upload edited data, missing independent reviews, missing native criteria, crossed
development/test families, injected file paths, or credentials in supporting records.
"""

import hashlib
import json

import pytest

from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS
from finetuning_labs.rewriter.benchmark import COMMON_SYSTEM, MODES, Case
from finetuning_labs.rewriter.evaluation import digest
from finetuning_labs.rewriter.reviewed_publication import audit_publication


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(root):
    save(
        root,
        "manifest.json",
        {
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            }
        },
    )


@pytest.fixture
def release(tmp_path):
    support, reviews = [], []
    for split, question in [
        ("train", "Can we meet?"),
        ("dev", "Is it ready?"),
        ("validation", "Will you call?"),
    ]:
        row = {
            "input": question.lower(),
            "output": question,
            "mode": "/slack",
            "checks": "Keep question.",
        }
        case = Case(id=split, **row)
        save(tmp_path, f"data/{split}.jsonl", row)
        save(
            tmp_path,
            f"sft/{split}.jsonl",
            {
                "system": COMMON_SYSTEM + MODES[case.mode],
                "user": case.user_message(),
                "assistant": case.output,
            },
        )
        review = {
            "id": split,
            "row_sha256": digest(row),
            "input_verdict": "pass",
            "output_verdict": "pass",
            "pair_verdict": "pass",
            "verdict": "pass",
            "critique": "The question is preserved.",
        }
        reviews.append(review)
        support.append(
            {
                "id": split,
                "row_sha256": digest(row),
                "split": split,
                "source_group": split,
                "context": {},
                "language": "en",
                "mechanical_issues": [],
                "human_reviewed": False,
                "opus_review": review | {"report": "reviews/one.json"},
                "decisions": {"labeling_engine." + k: {"score": 0.9} for k in DIMENSIONS},
            }
        )
    save(tmp_path, "supporting.json", support)
    save(tmp_path, "quality.json", {"splits": {"train": 1, "dev": 1, "validation": 1}})
    save(tmp_path, "reviews/one.json", {"reviews": reviews})
    (tmp_path / "README.md").write_text("Reviewed data. No human labels.\n")
    (tmp_path / "DATA_LICENSE.md").write_text("Private data. Preserve source licenses.\n")
    seal(tmp_path)
    return tmp_path


def test_publication_rejects_edited_file(release):
    (release / "data/train.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        audit_publication(release)


def test_publication_needs_the_actual_exact_review(release):
    save(release, "reviews/one.json", {"reviews": []})
    seal(release)
    with pytest.raises(ValueError, match="review"):
        audit_publication(release)


def test_publication_needs_all_native_decision_dimensions(release):
    meta = json.loads((release / "supporting.json").read_text())
    meta[0]["decisions"].pop("labeling_engine.pair_facts")
    save(release, "supporting.json", meta)
    seal(release)
    with pytest.raises(ValueError, match="decision"):
        audit_publication(release)


def test_dev_and_validation_cannot_share_source_family(release):
    meta = json.loads((release / "supporting.json").read_text())
    meta[2]["source_group"] = meta[1]["source_group"]
    save(release, "supporting.json", meta)
    seal(release)
    with pytest.raises(ValueError, match="source group"):
        audit_publication(release)


def test_publication_refuses_paths_outside_release(release):
    manifest = json.loads((release / "manifest.json").read_text())
    manifest["files"]["../private.txt"] = "bad"
    save(release, "manifest.json", manifest)
    with pytest.raises(ValueError, match="path"):
        audit_publication(release)


def test_publication_detects_secret_in_review_metadata(release):
    meta = json.loads((release / "supporting.json").read_text())
    meta[0]["note"] = "sk-or-v1-" + "a" * 64
    save(release, "supporting.json", meta)
    seal(release)
    with pytest.raises(ValueError, match="private"):
        audit_publication(release)
