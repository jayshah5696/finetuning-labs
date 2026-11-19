"""Current verification script: check exact expansion rows, anchors and code; optionally encode both pinned model presets.

The full native-record and raw-review audit is `just reviewed-audit`. These checks do not
establish semantic quality or replace the targeted review recorded with the release.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from finetuning_labs.rewriter.boost_data import validate_selection, verify_overlap_report
from finetuning_labs.rewriter.english_critical import (
    QUOTAS,
    critical_issues,
    verify_direction_balance,
)
from finetuning_labs.rewriter.evaluation import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/releases")
    args = parser.parse_args()
    source = ROOT / "data/rewriter/current"
    metadata = json.loads((source / "supporting.json").read_text())
    rows = [json.loads(line) for line in (source / "data/train.jsonl").read_text().splitlines()]
    train = [m for m in metadata if m["split"] == "train"]
    entries = [
        {"row": row, "meta": meta, "review": meta["data_review"]}
        for row, meta in zip(rows, train, strict=True)
        if meta.get("data_review", {}).get("report") == "reviews/sol-english-critical.json"
    ]
    validate_selection(entries, quotas=QUOTAS)
    evidence = source / "evidence/english-critical"
    verify_overlap_report(
        [e["row"] for e in entries], json.loads((evidence / "final-overlap-check.json").read_text())
    )
    balance = json.loads((evidence / "direction-balance.json").read_text())
    directions = verify_direction_balance(entries, balance)
    code_pairs, anchors = 0, 0

    def code_lines(text):
        outside = re.sub(r"```[^\n]*\n.*?\n```", "", text, flags=re.DOTALL)
        return {
            line.strip()
            for line in outside.splitlines()
            if re.match(
                r"\s*(?:def |class |import |from \w+ import |return |[A-Za-z_]\w*\s*=)", line
            )
        }

    for entry in entries:
        meta, row = entry["meta"], entry["row"]
        assert meta["language"] == "en" and meta["split"] == "train"
        assert not critical_issues(meta, row), meta["id"]
        for term in meta["critical_checks"].get("keep", []):
            anchors += 1
            assert re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", row["output"]), meta["id"]
        if "code" in meta["critical_checks"]:
            code_pairs += 1
            assert not code_lines(row["output"]) - code_lines(row["input"]), meta["id"]
    long = [e for e in entries if e["meta"]["length"] == "long"]
    assert len(long) == 100 and sum(e["meta"]["identity"] for e in entries) == 125
    assert all(1000 <= e["meta"]["tokens"]["input_tokens"] <= 2000 for e in long)
    excluded = json.loads((evidence / "quality-backfills.json").read_text())["replacements"]
    ids = {m["id"] for m in metadata}
    assert all(v["excluded_id"] not in ids and v["replacement_id"] in ids for v in excluded)
    write_json(
        args.output_dir / "english-expansion-check.json",
        {
            "selected_rows_sha256": digest([e["row"] for e in entries]),
            "rows": len(entries),
            "english_only": True,
            "quotas": QUOTAS,
            "identities": 125,
            "long_inputs": len(long),
            "long_input_token_range": [
                min(e["meta"]["tokens"]["input_tokens"] for e in long),
                max(e["meta"]["tokens"]["input_tokens"] for e in long),
            ],
            "whole_word_anchors_checked": anchors,
            "supplied_code_pairs_checked": code_pairs,
            "issues": [],
            "direction_counts": directions,
            "fresh_reverse_replacements": len(balance["replacements"]),
            "quality_backfills": excluded,
            "new_to_new_overlap": 0,
            "exact_pair_verification_passes": True,
            "human_labels": False,
        },
    )
    if not args.tokenizers:
        return
    from transformers import AutoTokenizer

    from finetuning_labs.rewriter.lora_training import encode_sft

    splits = {}
    for split in ["train", "dev", "validation"]:
        guides = [
            json.loads(line) for line in (source / f"sft/{split}.jsonl").read_text().splitlines()
        ]
        support = [m for m in metadata if m["split"] == split]
        splits[split] = [r for r, m in zip(guides, support, strict=True) if m["language"] == "en"]
    assert {s: len(r) for s, r in splits.items()} == {"train": 1979, "dev": 89, "validation": 141}
    previous, proofs = None, []
    for cfg in json.loads((ROOT / "configs/training-models.json").read_text()).values():
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], revision=cfg["model_revision"], token=False
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        encodings = []
        for rows in splits.values():
            for row in rows:
                encoded = encode_sft(row, tokenizer, 4096)
                target = [t for t in encoded["labels"] if t != -100]
                assert target[-1] == tokenizer.eos_token_id
                assert tokenizer.decode(target[:-1]) == row["assistant"]
                encodings.append(encoded)
        if previous is not None:
            assert encodings == previous
        previous = encodings
        proofs.append(
            {
                "model_id": cfg["model_id"],
                "revision": cfg["model_revision"],
                "rows": len(encodings),
                "max_chat_tokens": max(len(e["input_ids"]) for e in encodings),
                "assistant_targets_decode_exactly": True,
                "encoded_rows_sha256": digest(encodings),
            }
        )
    write_json(
        args.output_dir / "english-tokenizer-check.json",
        {
            "source_sft_file_sha256": {
                s: hashlib.sha256((source / f"sft/{s}.jsonl").read_bytes()).hexdigest()
                for s in splits
            },
            "english_sft_rows_sha256": {s: digest(r) for s, r in splits.items()},
            "english_split_counts": {s: len(r) for s, r in splits.items()},
            "models": proofs,
            "both_presets_produce_identical_encodings": True,
            "student_weights_loaded": False,
            "training_or_upload_performed": False,
        },
    )
    print(json.dumps({"rows": sum(len(r) for r in splits.values()), "models_checked": len(proofs)}))


if __name__ == "__main__":
    main()
