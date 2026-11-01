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
    assert [case.id for case in english.cases] == freeze["validation"]["case_ids"][:81]
    assert english.cases == tuple(case for case in full.cases if case.language == "en")
    assert len(full.cases) == 84
    assert len(english.cases) == 81
    assert english.manifest == full.manifest


def test_default_evaluation_keeps_non_english_cases():
    path = Path(__file__).resolve().parents[1] / "data/rewriter/reviewed-v2/validation-bundle"
    assert evaluation_bundle(path) == load_bundle(path)


@pytest.mark.parametrize("source,count", [("reviewed-v2", 81), ("current", 141)])
def test_modal_validation_requires_the_exact_published_sft_rows(source, count):
    from finetuning_labs.rewriter.benchmark import COMMON_SYSTEM, MODES
    from finetuning_labs.rewriter.evaluation import verify_published_validation

    root = Path(__file__).resolve().parents[1]
    bundle = evaluation_bundle(
        root / f"data/rewriter/{source}/validation-bundle", english_only=True
    )
    rows = [
        {"system": COMMON_SYSTEM + MODES[c.mode], "user": c.user_message(), "assistant": c.output}
        for c in bundle.cases
    ]
    assert len(rows) == count
    verify_published_validation(bundle, rows)
    for changed in [
        rows[:-1],
        list(reversed(rows)),
        rows[:-1] + [rows[-1] | {"assistant": "other"}],
    ]:
        with pytest.raises(ValueError, match="validation"):
            verify_published_validation(bundle, changed)


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
