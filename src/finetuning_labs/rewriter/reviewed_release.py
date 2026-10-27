"""Build releases only from exact rows independently verified by Opus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .benchmark import COMMON_SYSTEM, MODES, Case, Rules, deterministic_checks
from .evaluation import digest
from .training import Example, _tokenizer, token_audit, validate_splits, validation_index


def load_reviews(paths: list[Path]) -> dict:
    reviews = {}
    for path in paths:
        report = json.loads(path.read_text())
        for review in report["reviews"]:
            dimensions = [review[k] for k in ["input_verdict", "output_verdict", "pair_verdict"]]
            if any(x not in {"pass", "fail", "uncertain"} for x in dimensions):
                raise ValueError("invalid Opus dimension verdict")
            expected = (
                "fail"
                if "fail" in dimensions
                else "uncertain"
                if "uncertain" in dimensions
                else "pass"
            )
            if review["verdict"] != expected:
                raise ValueError("overall Opus verdict contradicts its dimensions")
            if not review["critique"].strip():
                raise ValueError("Opus critique is required")
            key = (review["id"], review["row_sha256"])
            if (
                key in reviews
                and {k: v for k, v in reviews[key].items() if k != "report"} != review
            ):
                raise ValueError("conflicting reviews for the same row")
            reviews[key] = review | {"report": str(path)}
    return reviews


def select_reviewed(pool: list[tuple[dict, dict]], reviews: dict):
    selected, excluded = [], []
    for row, meta in pool:
        sha = digest(row)
        if meta["row_sha256"] != sha:
            raise ValueError("row identity differs from its supporting metadata")
        review = reviews.get((meta["id"], sha))
        if meta["mechanical_issues"]:
            reason = "mechanical_failure"
        elif not review:
            reason = "unreviewed_exact_row"
        elif review["verdict"] != "pass":
            reason = review["verdict"]
        else:
            selected.append((row, meta | {"opus_review": review}))
            continue
        excluded.append({"id": meta["id"], "row_sha256": sha, "reason": reason})
    return selected, excluded


def export_release(pool_root: Path, review_paths: list[Path], destination: Path):
    if destination.exists():
        raise FileExistsError(destination)
    rows = [json.loads(x) for x in (pool_root / "candidates.jsonl").read_text().splitlines()]
    metadata = json.loads((pool_root / "scored-supporting.json").read_text())
    pool = list(zip(rows, metadata, strict=True))
    reviews = load_reviews(review_paths)
    selected, excluded = select_reviewed(pool, reviews)
    if not selected:
        raise ValueError("no independently reviewed rows")
    tokenizer, blocked = _tokenizer(), validation_index()
    examples, support = [], []
    identities, targets = set(), {}
    for row, meta in selected:
        if meta["id"] in identities:
            raise ValueError("duplicate release ID")
        identities.add(meta["id"])
        raw_rules = meta.get("rules", {})
        rules = Rules(
            required_any=tuple(tuple(x) for x in raw_rules.get("required_any", [])),
            forbidden=tuple(raw_rules.get("forbidden", [])),
            literals=tuple(raw_rules.get("literals", [])),
            identity=meta.get("identity", False),
        )
        case = Case(
            id=meta["id"], **row, context=meta["context"], language=meta["language"], rules=rules
        )
        failures = deterministic_checks(case, row["output"])
        if failures:
            raise ValueError(f"release reference contradicts literal rules: {case.id}: {failures}")
        if meta["split"] in {"train", "dev"} and (
            blocked.matches(case.input) or blocked.matches(case.output)
        ):
            raise ValueError("benchmark or personal holdout leakage")
        example = Example(
            case,
            "train" if meta["split"] == "train" else "validation",
            meta["source_group"],
            meta["length"],
            "independent",
            meta["id"],
        )
        tokens = token_audit(example, tokenizer)
        examples.append(example)
        targets.setdefault(meta["split"], set()).add(case.output)
        review = dict(meta["opus_review"])
        review["report"] = "reviews/" + Path(review["report"]).name
        support.append(
            {
                **meta,
                "opus_review": review,
                "human_reviewed": False,
                "review_status": "opus_verified_machine",
                "rules": asdict(rules),
                "tokens": {k: v for k, v in tokens.items() if k not in {"labels", "input_ids"}},
            }
        )
    validate_splits(examples)
    destination.mkdir(parents=True)
    (destination / "data").mkdir()
    (destination / "sft").mkdir()
    for split in ["train", "dev", "validation"]:
        chosen = [(r, m) for r, m in selected if m["split"] == split]
        data = [r for r, m in chosen]
        guides = []
        for row, meta in chosen:
            case = Case(id=meta["id"], **row, context=meta["context"], language=meta["language"])
            guides.append(
                {
                    "system": COMMON_SYSTEM + MODES[case.mode],
                    "user": case.user_message(),
                    "assistant": case.output,
                }
            )
        for name, values in [(f"data/{split}.jsonl", data), (f"sft/{split}.jsonl", guides)]:
            (destination / name).write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values)
            )
    (destination / "supporting.json").write_text(
        json.dumps(support, ensure_ascii=False, indent=2) + "\n"
    )
    (destination / "excluded.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2) + "\n"
    )
    long_inputs = [x["tokens"]["input_tokens"] for x in support if x["length"] == "long"]
    report = {
        "status": "opus_verified_synthetic_release",
        "splits": dict(Counter(x["split"] for x in support)),
        "by_mode": {
            split: dict(Counter(r["mode"] for r, m in selected if m["split"] == split))
            for split in ["train", "dev", "validation"]
        },
        "independent_targets": {k: len(v) for k, v in targets.items()},
        "source_groups": {
            split: len({x["source_group"] for x in support if x["split"] == split})
            for split in ["train", "dev", "validation"]
        },
        "languages": dict(Counter(x["language"] for x in support)),
        "long_pairs": len(long_inputs),
        "long_input_token_range": [min(long_inputs), max(long_inputs)] if long_inputs else [],
        "identity_by_length": dict(Counter(x["length"] for x in support if x.get("identity"))),
        "max_chat_tokens": max(x["tokens"]["chat_tokens"] for x in support),
        "max_reserved_inference_tokens": max(x["tokens"]["prompt_tokens"] + 2048 for x in support),
        "excluded_rows": len(excluded),
        "all_exact_rows_opus_pass": True,
        "human_reviewed": False,
        "independent_human_calibration": False,
        "original_baseline_modified": False,
        "locked_final_test": False,
    }
    (destination / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    hashes = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file()
    }
    (destination / "manifest.json").write_text(
        json.dumps({"role": "reviewed_rewriter_v2", "files": hashes}, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_release(args.pool, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
