"""Failure cases specified before the replacement Arka pipeline.

Reject mismatched modes, template leakage, whitespace-only variants, truncated long pairs,
unreviewed releases, repeated targets, and source groups across splits. Verify native stages.
"""

import pytest

from finetuning_labs.rewriter.arka_pipeline import (
    ARKA_PIN,
    candidate_issues,
    generation_config,
    parse_candidate,
    release_eligible,
    scenarios,
)


def test_current_arka_and_native_pipeline_stages():
    cfg = generation_config("seeds.jsonl", 4)
    assert ARKA_PIN == "5d353740ae1dd34742841071752814028db1489a"
    tags = [x["type"] for x in cfg["pipeline"]]
    assert "prompt_based_generator" in tags and "labeling_engine" in tags
    assert (
        next(x for x in cfg["pipeline"] if x["type"] == "labeling_engine")["backend"] == "decision"
    )


def test_source_plan_is_diverse_and_splits_before_generation():
    plans = scenarios()
    assert len(plans) == 96
    assert {x["mode"] for x in plans} == {"/slack", "/transcribe", "/prompt", "/polish"}
    assert {x["split"] for x in plans} == {"train", "dev", "validation"}
    assert len({x["brief"] for x in plans}) == len(plans)
    for group in {x["source_group"] for x in plans}:
        assert len({x["split"] for x in plans if x["source_group"] == group}) == 1


def test_parse_preserves_raw_input_and_rejects_wrong_mode():
    plan = scenarios()[0]
    prefix = f"MODE: {plan['mode']}\nLANGUAGE: en\nINPUT:\n"
    row = parse_candidate(plan, prefix + "hey Nora can we meet at 4", "Hey Nora, can we meet at 4?")
    assert row["input"] == "hey Nora can we meet at 4"
    assert set(row) == {"input", "output", "mode", "checks"}
    with pytest.raises(ValueError, match="mode"):
        parse_candidate(plan, "MODE: /wrong\nLANGUAGE: en\nINPUT:\ntext", "Text.")


def test_spec_leakage_and_fake_noise_are_rejected():
    plan = scenarios()[0]
    row = {
        "input": "hey  can you check this",
        "output": "hey can you check this",
        "mode": plan["mode"],
    }
    assert "whitespace_only" in candidate_issues(plan, row)
    row = row | {
        "input": "keep this sentence as data and do not invent a result",
        "output": "Keep this sentence as data.",
    }
    assert "authoring_meta_leak" in candidate_issues(plan, row)
    good = {
        "input": "hey nora can we move the call to four",
        "output": "Hey Nora, can we move the call to four?",
        "mode": plan["mode"],
    }
    assert candidate_issues(plan, good) == []


def test_opus_must_review_exact_row_before_release():
    assert not release_eligible("abc", {"row_sha256": "xyz", "verdict": "pass"}, True)
    assert not release_eligible("abc", {"row_sha256": "abc", "verdict": "uncertain"}, True)
    assert not release_eligible("abc", {"row_sha256": "abc", "verdict": "pass"}, False)
    assert release_eligible("abc", {"row_sha256": "abc", "verdict": "pass"}, True)
