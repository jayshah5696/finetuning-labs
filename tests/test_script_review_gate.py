"""Specify the user's Sol-only approval gate before implementing it."""

import hashlib

import pytest

from finetuning_labs.rewriter.reviewed_publication import require_script_review


def review(**changes):
    return {
        "reviewer_model": "gpt-6.1-sol",
        "verdict": "pass",
        "script_sha256": hashlib.sha256(b"script").hexdigest(),
        "notebook_sha256": hashlib.sha256(b"notebook").hexdigest(),
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"reviewer_model": "anthropic/claude-opus-5.5"},
        {"verdict": "fail"},
        {"script_sha256": "0" * 64},
        {"notebook_sha256": "0" * 64},
    ],
)
def test_stale_failed_or_wrong_model_review_cannot_approve_training(changes):
    with pytest.raises(ValueError, match="Sol"):
        require_script_review(review(**changes), b"script", b"notebook")


def test_exact_sol_pass_is_accepted():
    require_script_review(review(), b"script", b"notebook")
