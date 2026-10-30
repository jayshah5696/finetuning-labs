"""Reject unpinned adapters and comparisons that change more than the adapter."""

from pathlib import Path

import pytest

from finetuning_labs.rewriter.benchmark import load_bundle
from finetuning_labs.rewriter.evaluation import adapter_identity, evaluation_bundle


def test_english_evaluation_preserves_frozen_cases_and_order():
    import json

    root = Path(__file__).resolve().parents[1]
    path = root / "data/rewriter/reviewed-v2/validation-bundle"
    full = load_bundle(path)
    english = evaluation_bundle(path, english_only=True)
    freeze = json.loads((root / "data/rewriter/english-holdout-freeze.json").read_text())
    assert [case.id for case in english.cases] == freeze["validation"]["case_ids"]
    assert english.cases == tuple(case for case in full.cases if case.language == "en")
    assert len(full.cases) == 84
    assert len(english.cases) == 81
    assert english.manifest == full.manifest


def test_default_evaluation_keeps_non_english_cases():
    path = Path(__file__).resolve().parents[1] / "data/rewriter/reviewed-v2/validation-bundle"
    assert evaluation_bundle(path) == load_bundle(path)


def test_stock_and_pinned_adapter_are_distinct():
    assert adapter_identity({"model": "base"}) is None
    assert adapter_identity({"adapter_model": "owner/lora", "adapter_revision": "a" * 40}) == {
        "model": "owner/lora",
        "revision": "a" * 40,
    }


@pytest.mark.parametrize(
    "config",
    [
        {"adapter_model": "owner/lora"},
        {"adapter_revision": "a" * 40},
        {"adapter_model": "owner/lora", "adapter_revision": "main"},
    ],
)
def test_incomplete_or_mutable_adapter_identity_is_rejected(config):
    with pytest.raises(ValueError, match="adapter"):
        adapter_identity(config)
