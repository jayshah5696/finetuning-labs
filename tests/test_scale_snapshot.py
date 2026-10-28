"""Specify lineage failures before adding incremental verification of completed Arka calls."""

import json

import pytest

from finetuning_labs.rewriter.scale_data import decode_generated, expanded_plans


def test_completed_response_must_belong_to_the_expected_seed():
    raw = {"seed_id": "wrong", "generated_text": "{}"}
    with pytest.raises(ValueError, match="seed"):
        decode_generated(raw, expanded_plans()[0], "expected")


def test_source_header_cannot_be_reassigned_to_another_mode():
    raw = {
        "seed_id": "expected",
        "generated_text": json.dumps(
            {
                "instruction": "MODE: /polish\nLANGUAGE: en\nINPUT:\nhi",
                "response": "Hi.",
            }
        ),
    }
    with pytest.raises(ValueError, match="mode"):
        decode_generated(raw, expanded_plans()[0], "expected")
