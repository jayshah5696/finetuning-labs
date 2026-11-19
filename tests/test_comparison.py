"""Comparison failures specified before the paired-report implementation."""

import json

import pytest

from finetuning_labs.rewriter.benchmark import Bundle, Case, Rules
from finetuning_labs.rewriter.comparison import paired_report
from finetuning_labs.rewriter.evaluation import digest, generate_cases, judge_cases


def prepare(tmp_path, *, different_judge=False, generation_configs=None):
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
        config = generation_configs[name] if generation_configs else {"model": name}
        generate_cases(bundle, generation_dir, config, generate)

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


def test_adapter_comparison_rejects_different_generation_settings(tmp_path):
    stock = {"model": "base", "revision": "a" * 40, "max_new_tokens": 64}
    tuned = stock | {
        "adapter_model": "owner/lora",
        "adapter_revision": "b" * 40,
        "max_new_tokens": 128,
    }
    bundle = prepare(tmp_path, generation_configs={"left": stock, "right": tuned})
    with pytest.raises(ValueError, match="generation settings"):
        compare(bundle, tmp_path)


def test_adapter_comparison_also_checks_left_adapter_settings(tmp_path):
    stock = {"model": "base", "revision": "a" * 40, "max_new_tokens": 64}
    tuned = stock | {
        "adapter_model": "owner/lora",
        "adapter_revision": "b" * 40,
        "max_new_tokens": 128,
    }
    bundle = prepare(tmp_path, generation_configs={"left": tuned, "right": stock})
    with pytest.raises(ValueError, match="generation settings"):
        compare(bundle, tmp_path)


def test_usage_ignores_boolean_cost_and_token_fields():
    from finetuning_labs.rewriter.comparison import _usage

    records = [
        {
            "status": "completed",
            "result": {
                "usage": {
                    "input_tokens": True,
                    "output_tokens": 10,
                    "cost": True,
                    "cost_details": {"upstream_inference_cost": False},
                    "is_byok": True,
                }
            },
        }
    ]
    summary = _usage(records)
    assert summary["gateway_reported_cost_usd"] is None
    assert summary["upstream_reported_cost_usd"] is None
    assert summary["upstream_cost_coverage"] == 0
    assert summary["input_tokens"] == 0


def test_markdown_handles_none_pooled_gain():
    from finetuning_labs.rewriter.comparison import markdown

    empty_stats = {
        "scheduled": 0,
        "counts": dict.fromkeys(
            [
                "pass",
                "fail",
                "uncertain",
                "judge_error",
                "generation_error",
                "truncated",
                "pending",
            ],
            0,
        ),
        "verified_pass_rate_pct": None,
        "decided_case_pass_rate_pct": None,
        "assessment_coverage_pct": None,
        "critical_error_count": 0,
        "critical_error_rate_pct": None,
        "semantic_reviewed": 0,
        "semantic_review_coverage_pct": None,
        "identity_count": 0,
        "identity_unchanged": 0,
        "identity_rate_pct": None,
        "judge_error_events": 0,
    }
    side = {
        "pooled": empty_stats,
        "headline_equal_mode_pass_rate_pct": None,
        "per_mode": {},
        "by_length": {"short": empty_stats, "long": empty_stats},
    }
    report = {
        "scheduled_pairs": 0,
        "left_model": "left",
        "right_model": "right",
        "judge_config": {"model": "judge"},
        "left": side,
        "right": side,
        "pairs": {
            "both_pass": 0,
            "right_only_pass": 0,
            "left_only_pass": 0,
            "neither_verified_pass": 0,
        },
        "pooled_gain_percentage_points": None,
        "assessments": {"left": [], "right": []},
    }
    rendered = markdown(report)
    assert "N/A" in rendered


def test_comparison_cli_handles_io_errors_and_does_not_hide_type_errors(monkeypatch, tmp_path):
    from finetuning_labs.rewriter import comparison

    monkeypatch.setattr("sys.argv", ["comparison", "--run", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as exc_info:
        comparison.main()
    assert exc_info.value.code == 1

    monkeypatch.setattr(
        comparison,
        "evaluation_bundle",
        lambda *_, **__: (_ for _ in ()).throw(TypeError("internal-bug")),
    )
    with pytest.raises(TypeError, match="internal-bug"):
        comparison.main()
