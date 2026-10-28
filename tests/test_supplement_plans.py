"""Specify independent top-up sources before adding a supplemental generator."""

from collections import Counter

from finetuning_labs.rewriter.scale_data import expanded_plans, supplemental_plans


def test_topup_does_not_reuse_sources_or_create_validation_examples():
    old = expanded_plans()
    new = supplemental_plans()
    assert len(new) == len({p["id"] for p in new}) == 200
    assert {p["source_group"] for p in new}.isdisjoint(p["source_group"] for p in old)
    assert {p["domain"] for p in new}.isdisjoint(p["domain"] for p in old)
    assert Counter((p["split"], p["mode"]) for p in new) == {
        ("train", "/transcribe"): 120,
        ("dev", "/slack"): 20,
        ("dev", "/transcribe"): 20,
        ("dev", "/prompt"): 20,
        ("dev", "/polish"): 20,
    }
