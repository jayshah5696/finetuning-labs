"""Update and verify freeze manifests for rewriter benchmarks and holdouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .benchmark import load_bundle
from .evaluation import digest, write_json
from .hf_data import english_splits

PROJECT = Path(__file__).resolve().parents[3]


def update_holdout_freeze(project: Path = PROJECT) -> dict[str, str]:
    """Freeze current dev and validation data/SFT file bytes."""
    release = project / "data/rewriter/current"
    freeze = {
        name: hashlib.sha256((release / name).read_bytes()).hexdigest()
        for name in [
            "data/dev.jsonl",
            "data/validation.jsonl",
            "sft/dev.jsonl",
            "sft/validation.jsonl",
        ]
    }
    target = project / "data/rewriter/holdout-freeze.json"
    write_json(target, freeze)
    return freeze


def update_english_holdout_freeze(project: Path = PROJECT) -> dict:
    """Freeze current English holdouts (dev and validation) text, order, and case IDs."""
    release = project / "data/rewriter/current"
    rows = {
        s: [json.loads(x) for x in (release / f"data/{s}.jsonl").read_text().splitlines()]
        for s in ["dev", "validation"]
    }
    guides = {
        s: [json.loads(x) for x in (release / f"sft/{s}.jsonl").read_text().splitlines()]
        for s in ["dev", "validation"]
    }
    support = json.loads((release / "supporting.json").read_text())
    en_rows, en_guides = english_splits(rows, guides, support)

    freeze = {}
    for split in ["dev", "validation"]:
        ids = [m["id"] for m in support if m["split"] == split and m["language"] == "en"]
        freeze[split] = {
            "count": len(en_rows[split]),
            "data_sha256": digest(en_rows[split]),
            "sft_sha256": digest(en_guides[split]),
            "case_ids": ids,
        }
    target = project / "data/rewriter/english-holdout-freeze.json"
    write_json(target, freeze)
    return freeze


def update_validation_freeze(project: Path = PROJECT) -> dict:
    """Freeze calibration benchmark bundle (benchmark-draft) cases, fixtures, and hashes."""
    bundle_dir = project / "data/rewriter/benchmark-draft"
    bundle = load_bundle(bundle_dir)
    target = project / "data/rewriter/validation-freeze.json"
    existing = json.loads(target.read_text()) if target.exists() else {}

    files = {
        "cases.jsonl": hashlib.sha256((bundle_dir / "cases.jsonl").read_bytes()).hexdigest(),
        "supporting/cases.json": hashlib.sha256(
            (bundle_dir / "supporting/cases.json").read_bytes()
        ).hexdigest(),
        "supporting/fixtures.jsonl": hashlib.sha256(
            (bundle_dir / "supporting/fixtures.jsonl").read_bytes()
        ).hexdigest(),
        "supporting/judge-system.txt": hashlib.sha256(
            (bundle_dir / "supporting/judge-system.txt").read_bytes()
        ).hexdigest(),
    }

    cases_meta = json.loads((bundle_dir / "supporting/cases.json").read_text())
    source_groups = len({c.get("source_group", c["id"]) for c in cases_meta})

    freeze = {
        "role": existing.get("role", "development_validation"),
        "version": existing.get("version", "v1"),
        "accepted_date": existing.get("accepted_date", "2026-10-27"),
        "accepted_by": existing.get("accepted_by", "user"),
        "acceptance": existing.get(
            "acceptance", "I think this validation data is okay, we should use that"
        ),
        "bundle": "benchmark-draft",
        "cases": len(bundle.cases),
        "source_groups": source_groups,
        "files": files,
        "exclude_from_training": existing.get(
            "exclude_from_training",
            "All cases, reference outputs, fixtures, and related source families",
        ),
        "locked_final_test": existing.get("locked_final_test", False),
        "independent_human_output_labels": existing.get("independent_human_output_labels", False),
        "judge_human_calibrated": existing.get("judge_human_calibrated", False),
        "baseline_run": existing.get("baseline_run", "evals/baseline-20261027"),
        "note": existing.get(
            "note",
            "Dataset acceptance preserves the observed failures. It does not change per-case judgments or the pending human-label worksheet.",
        ),
    }
    write_json(target, freeze)
    return freeze


def update_all_freezes(project: Path = PROJECT) -> dict:
    """Update all freeze manifests together."""
    return {
        "holdout_freeze": update_holdout_freeze(project),
        "english_holdout_freeze": update_english_holdout_freeze(project),
        "validation_freeze": update_validation_freeze(project),
    }
