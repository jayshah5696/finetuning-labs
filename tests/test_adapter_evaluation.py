"""Reject unpinned adapters and comparisons that change more than the adapter."""

import pytest

from finetuning_labs.rewriter.evaluation import adapter_identity


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
