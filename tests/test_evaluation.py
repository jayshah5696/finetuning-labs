"""Failures specified before the model runner and publication code."""

import json
from pathlib import Path

import pytest

from finetuning_labs.rewriter.benchmark import Bundle, Case, Rules
from finetuning_labs.rewriter.evaluation import (
    _load_generations,
    aggregate,
    assess,
    generate_cases,
    judge_cases,
    saved_call,
)
from finetuning_labs.rewriter.hub import package_dataset


def sample(case_id="one", mode="/slack"):
    return Case(
        id=case_id,
        input="ask Rina to review 3 files",
        output="Ask Rina to review 3 files.",
        mode=mode,
        checks="Keep the recipient and quantity.",
        rules=Rules(required_any=(("Rina",), ("3",))),
    )


def review(case_id="one", **changes):
    return {
        "case_id": case_id,
        "verdict": "pass",
        "critique": "The recipient, quantity, and requested action are preserved.",
        "critical_errors": [],
        "criteria": dict.fromkeys(
            ["meaning", "terms_and_literals", "mode", "readability", "output_discipline"],
            "pass",
        ),
    } | changes


def generation(case_id="one", **changes):
    return {
        "case_id": case_id,
        "status": "completed",
        "text": "Ask Rina to review 3 files.",
        "finish_reason": "stop",
    } | changes


def test_missing_or_uncertain_judge_never_proves_a_pass():
    c = sample()
    assert assess(c, generation(), None)["status"] == "pending"
    uncertain = review(
        verdict="uncertain",
        criteria=review()["criteria"] | {"meaning": "uncertain"},
    )
    assert assess(c, generation(), uncertain)["status"] == "uncertain"
    assert assess(c, generation(), review())["status"] == "pass"


@pytest.mark.parametrize("status", ["generation_error", "truncated", "context_overflow"])
def test_execution_failure_cannot_be_rescued_by_a_passing_judge(status):
    result = assess(sample(), generation(status=status), review())
    assert result["status"] in {"generation_error", "truncated"}


def test_empty_output_invalid_review_and_wrong_case_fail_closed():
    assert assess(sample(), generation(text=""), review())["status"] == "generation_error"
    assert assess(sample(), generation(), review(case_id="another"))["status"] == "judge_error"
    wrong = generation(text="Ask Rinaldo to review 3 files.")
    assert assess(sample(), wrong, review())["status"] == "fail"
    with pytest.raises(ValueError, match="case"):
        assess(sample(), generation(case_id="another"), review())


def test_critical_error_is_not_averaged_away():
    bad = review(
        verdict="fail",
        critical_errors=["Changes the requested action."],
        criteria=review()["criteria"] | {"meaning": "fail"},
    )
    result = assess(sample(), generation(), bad)
    assert result["status"] == "fail" and result["critical_error"] is True


def test_metrics_keep_all_scheduled_cases_and_weight_modes_equally():
    cases = tuple(
        sample(str(n), mode)
        for n, mode in enumerate(
            ["/transcribe", "/transcribe", "/transcribe", "/slack", "/prompt", "/polish"]
        )
    )
    bundle = Bundle(cases=cases, fixtures=(), manifest={})
    results = [
        {
            "case_id": c.id,
            "status": "pass" if c.mode == "/transcribe" else "pending",
            "critical_error": False,
            "semantic_reviewed": c.mode == "/transcribe",
            "identity_required": False,
            "identity_unchanged": False,
        }
        for c in cases
    ]
    report = aggregate(bundle, results)
    assert report["pooled"]["verified_pass_rate_pct"] == 50
    assert report["headline_equal_mode_pass_rate_pct"] == 25
    assert report["pooled"]["assessment_coverage_pct"] == 50
    assert report["pooled"]["decided_case_pass_rate_pct"] == 100
    assert report["pooled"]["identity_rate_pct"] is None
    assert report["pooled"]["counts"]["pending"] == 3
    with pytest.raises(ValueError, match="case"):
        aggregate(bundle, results[:-1])


def test_saved_calls_resume_without_spend_and_reject_tampering(tmp_path):
    path = tmp_path / "call.json"
    calls = []

    def invoke():
        calls.append(1)
        return {"text": "answer"}

    assert saved_call(path, {"input": "one"}, invoke)["status"] == "completed"
    assert saved_call(path, {"input": "one"}, invoke)["result"] == {"text": "answer"}
    assert len(calls) == 1
    with pytest.raises(ValueError, match="request"):
        saved_call(path, {"input": "different"}, invoke)
    record = json.loads(path.read_text())
    record["result"]["text"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash"):
        saved_call(path, {"input": "one"}, invoke)


def test_failed_call_is_durable_redacted_and_not_implicitly_retried(tmp_path):
    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("secret-provider-key-must-not-appear")

    path = tmp_path / "failed.json"
    assert saved_call(path, {"input": "one"}, fail)["status"] == "error"
    assert "secret-provider-key" not in path.read_text()
    assert saved_call(path, {"input": "one"}, fail)["status"] == "error"
    assert len(calls) == 1


def test_generation_does_not_receive_the_reference_and_pins_run_identity(tmp_path):
    c = sample()
    bundle = Bundle(cases=(c,), fixtures=(), manifest={})
    observed = []

    def generate(messages):
        observed.append(messages)
        return {"text": c.output, "finish_reason": "stop", "model": "test-model"}

    config = {"model": "test-model", "max_new_tokens": 2048}
    generate_cases(bundle, tmp_path, config, generate)
    assert len(observed) == 1
    assert observed[0][1]["content"] == c.user_message()
    assert c.output not in observed[0][1]["content"]
    generate_cases(bundle, tmp_path, config, generate)
    assert len(observed) == 1
    with pytest.raises(ValueError, match="run"):
        generate_cases(bundle, tmp_path, config | {"model": "another"}, generate)


def test_judge_is_blind_and_result_binds_to_exact_candidate(tmp_path):
    c = sample()
    b = Bundle(cases=(c,), fixtures=(), manifest={})
    gen = tmp_path / "generation"
    generate_cases(
        b,
        gen,
        {"model": "test-model"},
        lambda _: {"text": c.output, "finish_reason": "stop", "model": "test-model"},
    )
    packets = []

    def judge(packet):
        packets.append(packet)
        return {"judgment": review(), "model": "test-judge"}

    judged = tmp_path / "judge"
    judge_cases(b, gen, judged, {"model": "test-judge"}, judge)
    assert "generator" not in packets[0] and "model" not in packets[0]
    assert packets[0]["candidate"] == c.output
    call = json.loads((gen / "calls/one.json").read_text())
    call["result"]["text"] = "Another candidate"
    (gen / "calls/one.json").write_text(json.dumps(call))
    with pytest.raises(ValueError, match="hash"):
        judge_cases(b, gen, judged, {"model": "test-judge"}, judge)


def test_generation_record_cannot_change_config_with_a_recomputed_hash(tmp_path):
    c = sample()
    bundle = Bundle(cases=(c,), fixtures=(), manifest={})
    generate_cases(
        bundle,
        tmp_path,
        {"model": "stock"},
        lambda _: {"text": c.output, "finish_reason": "stop"},
    )
    path = tmp_path / "calls/one.json"
    record = json.loads(path.read_text())
    record["request"]["config"]["model"] = "different-model"
    from finetuning_labs.rewriter.evaluation import digest

    record["request_sha256"] = digest(record["request"])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="request|config"):
        _load_generations(bundle, tmp_path)


def test_hub_package_is_an_allowlist_and_retains_simple_rows_and_license(tmp_path):
    import shutil

    source = Path(__file__).resolve().parents[1] / "data/rewriter/benchmark-draft"
    bundle = tmp_path / "bundle"
    shutil.copytree(source, bundle)
    (bundle / ".env").write_text("TOKEN=do-not-upload")
    (bundle / "personal").mkdir()
    (bundle / "personal/history.json").write_text("private-transcript-do-not-upload")
    output = tmp_path / "package"
    package_dataset(bundle, output, code_commit="test-commit")
    files = [p for p in output.rglob("*") if p.is_file()]
    assert not any(p.name == ".env" or "personal" in p.parts for p in files)
    assert not any(b"do-not-upload" in p.read_bytes() for p in files)
    rows = [json.loads(x) for x in (output / "data/calibration.jsonl").read_text().splitlines()]
    assert len(rows) == 128
    assert all(set(row) == {"input", "output", "mode", "checks"} for row in rows)
    assert (output / "bundle/supporting/TYPEWHISPER-LICENSE.txt").read_bytes() == (
        source / "supporting/TYPEWHISPER-LICENSE.txt"
    ).read_bytes()
    assert "pending" in (output / "README.md").read_text().lower()
    with pytest.raises(FileExistsError):
        package_dataset(bundle, output, code_commit="test-commit")


def test_generate_and_judge_reject_nonpositive_workers(tmp_path):
    c = sample()
    b = Bundle(cases=(c,), fixtures=(), manifest={})
    with pytest.raises(ValueError, match="workers"):
        generate_cases(b, tmp_path / "gen", {"model": "test"}, lambda _: {}, workers=0)
    with pytest.raises(ValueError, match="workers"):
        judge_cases(
            b, tmp_path / "gen", tmp_path / "judge", {"model": "test"}, lambda _: {}, workers=0
        )


def test_aggregate_rejects_empty_bundle_and_assessments():
    bundle = Bundle(cases=(), fixtures=(), manifest={})
    with pytest.raises(ValueError, match="empty|case"):
        aggregate(bundle, [])


def test_evaluation_cli_rejects_api_key_file_and_handles_missing_config(monkeypatch, tmp_path):
    from finetuning_labs.rewriter.evaluation import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation",
            "--api-key-file",
            str(tmp_path / "key"),
            "generate",
            "--model",
            "luna",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation",
            "--config",
            str(tmp_path / "missing.toml"),
            "generate",
            "--model",
            "luna",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    empty_cfg = tmp_path / "empty.toml"
    empty_cfg.write_text("")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation",
            "--config",
            str(empty_cfg),
            "generate",
            "--model",
            "luna",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    from finetuning_labs.rewriter import evaluation

    monkeypatch.setattr(
        evaluation, "evaluation_bundle", lambda *_, **__: (_ for _ in ()).throw(TypeError("bug"))
    )
    with pytest.raises(TypeError, match="bug"):
        main()
