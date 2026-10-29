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


def test_opaque_provider_ciphertext_does_not_trigger_plaintext_hf_token_scan(release):
    blob = "gAAAAhf_" + "a" * 40
    record = {
        "result": {
            "choices": [
                {
                    "message": {
                        "reasoning_details": [
                            {
                                "type": "reasoning.encrypted",
                                "format": "openai-responses-v1",
                                "data": blob,
                            }
                        ]
                    }
                }
            ]
        }
    }
    save(release, "evidence/provider.json", record)
    seal(release)
    assert audit_publication(release)
    with pytest.raises(ValueError, match="private"):
        audit_publication(release, private_needles=[blob.encode()])
    record["result"]["choices"][0]["message"]["reasoning_details"][0]["type"] = "reasoning.text"
    save(release, "evidence/provider.json", record)
    seal(release)
    with pytest.raises(ValueError, match="private"):
        audit_publication(release)


@pytest.mark.parametrize("name", ["notebooks/old.ipynb", "runtime/train_rewriter.py"])
def test_data_audit_does_not_require_retired_runtime_artifacts(release, name):
    manifest = json.loads((release / "manifest.json").read_text())
    manifest["files"][name] = hashlib.sha256(b"historical code").hexdigest()
    save(release, "manifest.json", manifest)
    verified = audit_publication(release)
    assert name not in verified
    assert json.loads(verified["data/train.jsonl"])["output"] == "Can we meet?"


@pytest.mark.parametrize("change", ["review", "native_export", "plan"])
@pytest.mark.parametrize("namespace", ["boost", "english-critical"])
def test_sol_verified_addition_must_bind_to_its_actual_provider_call(release, change, namespace):
    from finetuning_labs.rewriter.boost_data import SOL, review_packet
    from finetuning_labs.rewriter.evaluation import saved_call
    from finetuning_labs.rewriter.scale_data import VERIFY_SYSTEM

    meta = json.loads((release / "supporting.json").read_text())
    item = meta[0]
    row = json.loads((release / "data/train.jsonl").read_text())
    item.update(identity=False, native_accepted=True, protected_literals=[], record_id="native-one")
    for value in item["decisions"].values():
        value["confidence"] = 0.8
    saved = item.pop("opus_review")
    item["data_review"] = saved
    item["verification_calls"] = ["evidence/sol.json"]
    item["adjudication_review"] = None
    item["native_evidence"] = "evidence/native.json"
    item["native_exports"] = "evidence/native-exports.json"
    item["overlap_evidence"] = f"evidence/{namespace}/final-overlap-check.json"
    item["critical_checks"] = {"keep": ["Can"]}
    item["generation_plan"] = f"evidence/{namespace}/source-plan.json"
    save(
        release,
        item["generation_plan"],
        {"plans": [{"id": item["id"], "critical_checks": item["critical_checks"]}]},
    )
    report = json.loads((release / "reviews/one.json").read_text())
    report["reviewer_model"] = SOL
    save(release, "reviews/one.json", report)
    review = {k: v for k, v in saved.items() if k != "report"}
    request = {
        "model": SOL,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": json.dumps([review_packet(row, item)])},
        ],
    }
    result = {
        "model": SOL,
        "id": "unique-call",
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps({"reviews": [review]})}}
        ],
    }
    saved_call(release / "evidence/sol.json", request, lambda: result)
    native = {
        item["id"]: {
            "id": item["record_id"],
            "payload": {
                "instruction": Case(id=item["id"], **row).user_message(),
                "response": row["output"],
            },
            "scores": {
                "decisions": item["decisions"],
                "judge_model": "openai/gpt-6-luna-decisions",
            },
        }
    }
    save(release, "evidence/native.json", native)
    save(
        release,
        item["overlap_evidence"],
        {"rows": 1, "new_to_new_overlap": 0, "rows_sha256": digest([row])},
    )
    save(release, "evidence/native-exports.json", [digest(native[item["id"]]["payload"])])
    save(release, "supporting.json", meta)
    seal(release)
    assert "data/train.jsonl" in audit_publication(release)
    if change == "plan":
        item["critical_checks"] = {}
        save(release, "supporting.json", meta)
        seal(release)
        with pytest.raises(ValueError, match="plan"):
            audit_publication(release)
        return
    if change == "native_export":
        save(release, "evidence/native-exports.json", [])
        seal(release)
        with pytest.raises(ValueError, match="native|Native"):
            audit_publication(release)
        return
    result["choices"][0]["message"]["content"] = json.dumps(
        {"reviews": [review | {"pair_verdict": "fail", "verdict": "fail"}]}
    )
    record = json.loads((release / "evidence/sol.json").read_text())
    record["result"] = result
    record["result_sha256"] = digest(result)
    save(release, "evidence/sol.json", record)
    seal(release)
    with pytest.raises(ValueError, match="verification|review"):
        audit_publication(release)
