"""Specify the clean Hub layout and row-preservation failures before implementation."""

import io

import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from finetuning_labs.rewriter.hf_data import build_dataset, cleanup_paths


def example():
    row = {
        "input": 'hello\n1. keep "cache.enabled"',
        "output": 'Hello.\n1. Keep "cache.enabled".',
        "mode": "/transcribe",
        "checks": "Preserve the literal and both lines.",
    }
    guide = {
        "system": "Rewrite only.",
        "user": "MODE: /transcribe\nLANGUAGE: en\nINPUT:\n" + row["input"],
        "assistant": row["output"],
    }
    return row, guide


def build(row=None, guide=None):
    expected_row, expected_guide = example()
    return build_dataset(
        {"train": [row or expected_row]},
        {"train": [guide or expected_guide]},
        title="Rewriter",
        description="A small test.",
        license_text=b"Source notices.",
        source_license=b"GPL source notice.",
    )


def test_parquet_roundtrip_keeps_visible_and_supervised_fields_exactly():
    files = build()
    row, guide = example()
    assert pq.read_table(io.BytesIO(files["data/train.parquet"])).to_pylist() == [row]
    assert pq.read_table(io.BytesIO(files["sft/train.parquet"])).to_pylist() == [guide]
    assert set(files) == {
        "README.md",
        "DATA_LICENSE.md",
        "licenses/TYPEWHISPER-LICENSE.txt",
        "data/train.parquet",
        "sft/train.parquet",
    }
    assert b"default: true" in files["README.md"]


@pytest.mark.parametrize("change", ["metadata", "wrong_target", "missing_context"])
def test_extra_metadata_or_changed_supervision_is_rejected(change):
    row, guide = example()
    if change == "metadata":
        row["review"] = "pass"
    elif change == "wrong_target":
        guide["assistant"] = "An unrelated answer."
    else:
        guide["user"] = "Different input."
    with pytest.raises(ValueError):
        build(row, guide)


def test_cleanup_removes_code_reports_and_old_data_paths():
    files = build()
    existing = {
        ".gitattributes",
        "README.md",
        "runtime/train.py",
        "notebooks/train.ipynb",
        "audit/reviews.json",
        "old/train.jsonl",
        "data/train.parquet",
    }
    assert cleanup_paths(existing, files) == [
        "audit/reviews.json",
        "notebooks/train.ipynb",
        "old/train.jsonl",
        "runtime/train.py",
    ]
    with pytest.raises(ValueError, match="data"):
        cleanup_paths(existing, {**files, "runtime/train.py": b"code"})
