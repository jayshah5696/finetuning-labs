"""Comparison failures specified before the paired-report implementation."""

import json

import pytest

from finetuning_labs.rewriter.benchmark import Bundle, Case, Rules
from finetuning_labs.rewriter.comparison import paired_report
from finetuning_labs.rewriter.evaluation import digest, generate_cases, judge_cases


def prepare(tmp_path, *, different_judge=False):
    cases = (
        Case(
            "one",
            "ask Rina to review 3 files",
            "Ask Rina to review 3 files.",
            "/slack",
            "Preserve the recipient and count.",
            Rules(required_any=(("Rina",), ("3",))),
        ),
        Case(
            "two",
            "ask Rina to review 3 files",
            "Ask Rina to review 3 files.",
            "/prompt",
            "Preserve the recipient and count.",
            Rules(required_any=(("Rina",), ("3",))),
        ),
    )
    bundle = Bundle(cases=cases, fixtures=(), manifest={"long_form": {"case_ids": ["two"]}})
    for name in ("left", "right"):
        outputs = iter(
            [
                cases[0].output,
                "Ask Rinaldo to review 3 files." if name == "left" else cases[1].output,
            ]
        )

        def generate(_, selected_outputs=outputs, model_name=name):
            return {"text": next(selected_outputs), "model": model_name, "finish_reason": "stop"}

        generation_dir = tmp_path / name
        generate_cases(bundle, generation_dir, {"model": name}, generate)

        def judge(packet):
            wrong = "Rinaldo" in packet["candidate"]
            criteria = dict.fromkeys(
                ["meaning", "terms_and_literals", "mode", "readability", "output_discipline"],
                "pass",
            )
            if wrong:
                criteria["meaning"] = "fail"
            return {
                "judgment": {
                    "case_id": packet["case_id"],
                    "verdict": "fail" if wrong else "pass",
                    "critique": "The recipient changed." if wrong else "The request is preserved.",
                    "critical_errors": ["Changed recipient"] if wrong else [],
                    "criteria": criteria,
                }
            }

        config = {"model": "same-judge", "rubric": "same rubric"}
        if name == "right" and different_judge:
            config["rubric"] = "a different rubric"
        judge_cases(bundle, generation_dir, tmp_path / (name + "-judge"), config, judge)
    return bundle


def compare(bundle, tmp_path):
    return paired_report(
        bundle,
        tmp_path / "left",
        tmp_path / "right",
        tmp_path / "left-judge",
        tmp_path / "right-judge",
    )


def test_paired_counts_come_from_bound_calls_not_edited_summary_files(tmp_path):
    bundle = prepare(tmp_path)
    (tmp_path / "left-judge/metrics.json").write_text('{"fake_pass_rate": 100}')
    report = compare(bundle, tmp_path)
    assert report["left"]["pooled"]["verified_pass_rate_pct"] == 50
    assert report["right"]["pooled"]["verified_pass_rate_pct"] == 100
    assert report["pairs"] == {
        "both_pass": 1,
        "right_only_pass": 1,
        "left_only_pass": 0,
        "neither_verified_pass": 0,
    }
    assert report["pooled_gain_percentage_points"] == 50
    assert report["human_calibrated"] is False and report["locked_test"] is False


def test_comparison_rejects_changed_judge_or_rubric(tmp_path):
    bundle = prepare(tmp_path, different_judge=True)
    with pytest.raises(ValueError, match="judge"):
        compare(bundle, tmp_path)


def test_valid_hashes_cannot_bind_a_judgment_to_another_candidate(tmp_path):
    bundle = prepare(tmp_path)
    path = tmp_path / "right-judge/calls/one.json"
    record = json.loads(path.read_text())
    record["request"]["packet"]["candidate"] = "a foreign candidate"
    record["request_sha256"] = digest(record["request"])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="candidate|packet"):
        compare(bundle, tmp_path)
