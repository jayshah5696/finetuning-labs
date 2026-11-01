"""Failure tests written before the benchmark implementation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from finetuning_labs.rewriter.benchmark import (
    Bundle,
    Case,
    Rules,
    audit,
    deterministic_checks,
    export_guide,
    judge_packet,
    load_bundle,
    render_review,
    token_length_report,
    validate_judgment,
    verdict,
)


def case(**changes):
    values = {
        "id": "sample",
        "input": "ask Rina to review three files",
        "output": "Ask Rina to review three files.",
        "mode": "/slack",
        "checks": "Preserve the recipient and file count.",
        "rules": Rules(required_any=(("Rina",), ("three", "3"))),
        "context": {},
    }
    return Case(**(values | changes))


def judgment(**changes):
    return {
        "case_id": "sample",
        "verdict": "pass",
        "critique": "The recipient and file count are preserved.",
        "critical_errors": [],
        "criteria": {
            "meaning": "pass",
            "terms_and_literals": "pass",
            "mode": "pass",
            "readability": "pass",
            "output_discipline": "pass",
        },
    } | changes


def test_required_names_are_not_substrings_and_equivalents_are_explicit():
    assert deterministic_checks(case(), "Ask Rina to review 3 files.") == []
    assert "required: Rina" in deterministic_checks(case(), "Ask Rinaldo to review 3 files.")
    assert "required: three | 3" in deterministic_checks(case(), "Ask Rina to review 4 files.")


def test_forbidden_words_and_exact_literals_have_different_rules():
    sample = case(rules=Rules(forbidden=("Monday",), literals=("user_id",)))
    assert deterministic_checks(sample, "Keep `user_id` on Tuesday.") == []
    assert "forbidden: Monday" in deterministic_checks(sample, "Keep `user_id` on Monday.")
    assert "literal: user_id" in deterministic_checks(sample, "Keep `USER_ID` on Tuesday.")


def test_identity_is_only_required_when_declared():
    sample = case(input="Thanks, Rina!", output="Thanks, Rina!", rules=Rules(identity=True))
    assert deterministic_checks(sample, "Thanks, Rina!") == []
    assert "identity changed" in deterministic_checks(sample, "Thank you, Rina!")
    assert deterministic_checks(case(), "Could Rina review 3 files?") == []


def test_empty_output_and_literal_success_do_not_become_semantic_passes():
    assert "empty output" in deterministic_checks(case(), "  ")
    assert verdict(case(), "Ask Rina to review 3 files.", None) == "pending"
    assert verdict(case(), "Ask Rinaldo to review 3 files.", None) == "fail"


def test_changed_meaning_can_pass_terms_and_still_fail_semantic_review():
    sample = case(
        input="the invoice is $47.50 not $74.50",
        output="The invoice is $47.50, not $74.50.",
        rules=Rules(required_any=(("47.50",), ("74.50",))),
    )
    wrong = "The invoice is $74.50, not $47.50."
    assert deterministic_checks(sample, wrong) == []
    review = judgment(
        verdict="fail",
        critique="The candidate reverses which invoice amount is correct.",
        critical_errors=["Correct and incorrect amounts are reversed."],
        criteria=judgment()["criteria"] | {"meaning": "fail"},
    )
    assert verdict(sample, wrong, review) == "fail"


@pytest.mark.parametrize(
    "review",
    [
        judgment(case_id="another-case"),
        judgment(critique=""),
        judgment(criteria={"meaning": "pass"}),
        judgment(critical_errors=["Changed the recipient."]),
        judgment(criteria=judgment()["criteria"] | {"meaning": "fail"}),
        judgment(verdict="fail"),
    ],
)
def test_invalid_or_contradictory_judgments_are_rejected(review):
    with pytest.raises(ValueError):
        validate_judgment("sample", review)


def test_uncertain_remains_uncertain():
    review = judgment(
        verdict="uncertain",
        critique="The intended count is ambiguous in the supplied input.",
        criteria=judgment()["criteria"] | {"meaning": "uncertain"},
    )
    assert verdict(case(), "Ask Rina to review 3 files.", review) == "uncertain"


def test_prompt_context_is_available_but_review_metadata_is_not():
    sample = case(context={"profile": "developer", "dictionary": ["DictationOrchestrator.cs"]})
    packet = judge_packet(sample, "candidate")
    assert packet["context"] == sample.context
    assert "source" not in packet and "generator" not in packet
    assert "PROFILE: developer" in sample.user_message()
    assert "DICTIONARY: DictationOrchestrator.cs" in sample.user_message()
    assert "checks" not in sample.user_message().lower()


def test_review_escapes_case_html_and_export_refuses_overwrite(tmp_path):
    sample = case(input="<script>alert(1)</script>")
    bundle = Bundle(cases=(sample,), fixtures=(), manifest={})
    page = render_review(bundle)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>" not in page
    path = tmp_path / "guide.jsonl"
    export_guide(bundle, path)
    row = json.loads(path.read_text())
    assert set(row) == {"system", "user", "assistant"}
    assert row["assistant"] == sample.output
    with pytest.raises(FileExistsError):
        export_guide(bundle, path)


def test_actual_bundle_is_self_contained_and_all_modes_have_attacks():
    bundle = load_bundle()
    report = audit(bundle)
    assert len(bundle.cases) == 128
    assert report["source_counts"] == {
        "typewhisper_authored": 30,
        "new_authored": 38,
        "expansion_benchmark": 60,
    }
    assert set(report["modes"]) == {"/slack", "/transcribe", "/prompt", "/polish"}
    assert report["mechanical_failures"] == []
    assert report["semantic_judge_run"] is False
    assert all(report["fixtures_by_mode"][mode]["fail"] for mode in report["modes"])


def test_bundle_rejects_extra_row_metadata_duplicate_ids_and_source_changes(tmp_path):
    import shutil

    source = Path(__file__).resolve().parents[1] / "data/rewriter/benchmark-draft"
    copy = tmp_path / "bundle"
    shutil.copytree(source, copy)
    rows_path = copy / "cases.jsonl"
    original = rows_path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["teacher"] = "should not be in a visible row"
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="fields"):
        load_bundle(copy)
    rows_path.write_text(original)
    support = copy / "supporting/cases.json"
    data = json.loads(support.read_text())
    data[1]["id"] = data[0]["id"]
    support.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_bundle(copy)
    shutil.copy2(source / "supporting/cases.json", support)
    with (copy / "supporting/typewhisper-cases.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source snapshot"):
        load_bundle(copy)


def test_audit_rejects_reference_conflicts_and_duplicate_case_text():
    sample = case()
    conflict = replace(sample, rules=Rules(required_any=(("Ravi",),)))
    with pytest.raises(ValueError, match="reference"):
        audit(Bundle(cases=(conflict,), fixtures=(), manifest={}))
    with pytest.raises(ValueError, match="duplicate"):
        audit(Bundle(cases=(sample, replace(sample, id="different-id")), fixtures=(), manifest={}))


class CharacterTokenizer:
    """Count characters so tests can expose missing chat overhead without downloads."""

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=True):
        assert tokenize is True
        ids = [0] * (sum(len(message["content"]) for message in messages) + 12)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids


def test_lengths_count_the_entire_chat_and_reserve_output_without_truncation():
    sample = case(input="a" * 1800, output="b" * 1800)
    bundle = Bundle(cases=(sample,), fixtures=(), manifest={"long_form": {"case_ids": ["sample"]}})
    report = token_length_report(
        bundle, CharacterTokenizer(), max_seq_length=2000, max_new_tokens=2048
    )
    row = report["cases"][0]
    assert row["input_tokens"] == row["output_tokens"] == 1800
    assert row["prompt_tokens"] > 1800
    assert row["training_tokens"] == row["prompt_tokens"] + 1800
    assert row["reserved_inference_tokens"] == row["prompt_tokens"] + 2048
    assert row["training_fits"] is False and row["inference_fits"] is False
    assert report["input_bound_failures"] == []
    assert report["context_overflow_cases"] == ["sample"]


@pytest.mark.parametrize("size", [999, 2001])
def test_long_input_bounds_apply_to_input_not_combined_sequence(size):
    bundle = Bundle(
        cases=(case(input="a" * size, output="b" * 1500),),
        fixtures=(),
        manifest={"long_form": {"case_ids": ["sample"]}},
    )
    report = token_length_report(bundle, CharacterTokenizer())
    assert report["input_bound_failures"] == ["sample"]


def test_token_report_rejects_unknown_long_cases_and_invalid_limits():
    bundle = Bundle(cases=(case(),), fixtures=(), manifest={"long_form": {"case_ids": ["missing"]}})
    with pytest.raises(ValueError, match="unknown long"):
        token_length_report(bundle, CharacterTokenizer())
    with pytest.raises(ValueError, match="positive"):
        token_length_report(replace(bundle, manifest={}), CharacterTokenizer(), max_seq_length=0)


def test_long_review_text_is_complete_escaped_and_collapsed():
    text = '<quote>"Keep this."</quote>\n\n' + "Long paragraph. " * 100
    sample = case(input=text, output=text, rules=Rules())
    page = render_review(Bundle(cases=(sample,), fixtures=(), manifest={}))
    assert "Read full input" in page and "Read full expected output" in page
    assert "&lt;quote&gt;&quot;Keep this.&quot;&lt;/quote&gt;" in page
    assert "Long paragraph. " * 100 in page
    assert "<quote>" not in page
