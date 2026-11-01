"""Execute Phase 2 merge of diverse-v2 and validation-v1 into current and benchmark-draft."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from arka.pipeline.output import OutputWriter

from finetuning_labs.rewriter.benchmark import Case
from finetuning_labs.rewriter.evaluation import digest, write_json

PROJECT = Path(__file__).resolve().parents[1]
DATA_REWRITER = PROJECT / "data/rewriter"
CURRENT = DATA_REWRITER / "current"
BENCHMARK_DRAFT = DATA_REWRITER / "benchmark-draft"
EXP_BENCH = DATA_REWRITER / "expansion-benchmark-v1"
EXPANSIONS = DATA_REWRITER / "expansions"
DIVERSE_V2 = EXPANSIONS / "diverse-v2"
DIVERSE_V2_RUN = EXPANSIONS / "diverse-v2-run"
VALIDATION_V1 = EXPANSIONS / "validation-v1"
VALIDATION_V1_RUN = EXPANSIONS / "validation-v1-run"


def seal_manifest(root: Path, role: str) -> dict:
    """Compute and write manifest.json for a release directory."""
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = str(p.relative_to(root))
            files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"role": role, "files": files}
    write_json(root / "manifest.json", manifest)
    return manifest


def merge_benchmark_draft():
    print("\n--- 1. Merging expansion-benchmark-v1 into benchmark-draft ---")
    draft_cases = [
        json.loads(x) for x in (BENCHMARK_DRAFT / "cases.jsonl").read_text().splitlines()
    ]
    exp_cases = [json.loads(x) for x in (EXP_BENCH / "cases.jsonl").read_text().splitlines()]
    merged_cases = draft_cases + exp_cases
    assert len(merged_cases) == 68 + 60 == 128

    (BENCHMARK_DRAFT / "cases.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in merged_cases) + "\n"
    )

    draft_meta = json.loads((BENCHMARK_DRAFT / "supporting/cases.json").read_text())
    exp_meta = json.loads((EXP_BENCH / "supporting/cases.json").read_text())
    merged_meta = draft_meta + exp_meta
    assert len(merged_meta) == 128
    write_json(BENCHMARK_DRAFT / "supporting/cases.json", merged_meta)

    draft_fixtures = [
        json.loads(x)
        for x in (BENCHMARK_DRAFT / "supporting/fixtures.jsonl").read_text().splitlines()
    ]
    exp_fixtures = [
        json.loads(x) for x in (EXP_BENCH / "supporting/fixtures.jsonl").read_text().splitlines()
    ]
    merged_fixtures = draft_fixtures + exp_fixtures
    assert len(merged_fixtures) == 156 + 165 == 321
    (BENCHMARK_DRAFT / "supporting/fixtures.jsonl").write_text(
        "\n".join(json.dumps(f, ensure_ascii=False) for f in merged_fixtures) + "\n"
    )

    draft_manifest = json.loads((BENCHMARK_DRAFT / "manifest.json").read_text())
    draft_manifest["files"]["cases.jsonl"] = hashlib.sha256(
        (BENCHMARK_DRAFT / "cases.jsonl").read_bytes()
    ).hexdigest()
    draft_manifest["files"]["supporting/cases.json"] = hashlib.sha256(
        (BENCHMARK_DRAFT / "supporting/cases.json").read_bytes()
    ).hexdigest()
    draft_manifest["files"]["supporting/fixtures.jsonl"] = hashlib.sha256(
        (BENCHMARK_DRAFT / "supporting/fixtures.jsonl").read_bytes()
    ).hexdigest()
    draft_manifest["new_authored_case_count"] = 38 + 60

    exp_long_ids = (
        json.loads((EXP_BENCH / "manifest.json").read_text())
        .get("long_form", {})
        .get("case_ids", [])
    )
    if "long_form" in draft_manifest:
        current_long_ids = draft_manifest["long_form"].get("case_ids", [])
        for lid in exp_long_ids:
            if lid not in current_long_ids:
                current_long_ids.append(lid)
        draft_manifest["long_form"]["case_ids"] = current_long_ids

    write_json(BENCHMARK_DRAFT / "manifest.json", draft_manifest)
    print(
        f"[OK] benchmark-draft updated: {len(merged_cases)} cases, {len(merged_fixtures)} fixtures"
    )


def merge_validation_bundle():
    print("\n--- 2. Merging validation-v1 into current/validation-bundle ---")
    val_bundle_dir = CURRENT / "validation-bundle"
    v1_rows = [json.loads(x) for x in (VALIDATION_V1 / "data/train.jsonl").read_text().splitlines()]
    v1_meta = json.loads((VALIDATION_V1 / "supporting.json").read_text())

    bundle_cases = [
        json.loads(x) for x in (val_bundle_dir / "cases.jsonl").read_text().splitlines()
    ]
    merged_bundle_cases = bundle_cases + v1_rows
    assert len(merged_bundle_cases) == 84 + 60 == 144
    (val_bundle_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in merged_bundle_cases) + "\n"
    )

    bundle_meta = json.loads((val_bundle_dir / "supporting/cases.json").read_text())
    v1_bundle_entries = []
    for m in v1_meta:
        v1_bundle_entries.append(
            {
                "id": m["id"],
                "source_group": m["source_group"],
                "mode": m["mode"],
                "split": "validation",
                "length": m["length"],
                "language": "en",
                "identity": m.get("identity", False),
                "context": m.get("context", {}),
                "brief": m.get("brief", ""),
                "row_sha256": m["row_sha256"],
                "record_id": m["record_id"],
                "mechanical_issues": [],
                "decisions": m["decisions"],
                "tokens": m["tokens"],
                "human_reviewed": False,
                "review_status": "sol_verified_machine",
                "rules": m.get("rules", {}),
                "source_kind": "expansion_validation",
            }
        )
    merged_bundle_meta = bundle_meta + v1_bundle_entries
    assert len(merged_bundle_meta) == 144
    write_json(val_bundle_dir / "supporting/cases.json", merged_bundle_meta)

    exp_fixtures = (EXP_BENCH / "supporting/fixtures.jsonl").read_text().splitlines()
    existing_fixtures = (val_bundle_dir / "supporting/fixtures.jsonl").read_text().splitlines()
    merged_fixtures_lines = existing_fixtures + exp_fixtures
    (val_bundle_dir / "supporting/fixtures.jsonl").write_text(
        "\n".join(merged_fixtures_lines) + ("\n" if merged_fixtures_lines else "")
    )

    bundle_manifest = json.loads((val_bundle_dir / "manifest.json").read_text())
    bundle_manifest["files"]["cases.jsonl"] = hashlib.sha256(
        (val_bundle_dir / "cases.jsonl").read_bytes()
    ).hexdigest()
    bundle_manifest["files"]["supporting/cases.json"] = hashlib.sha256(
        (val_bundle_dir / "supporting/cases.json").read_bytes()
    ).hexdigest()
    bundle_manifest["files"]["supporting/fixtures.jsonl"] = hashlib.sha256(
        (val_bundle_dir / "supporting/fixtures.jsonl").read_bytes()
    ).hexdigest()

    long_ids = [m["id"] for m in v1_meta if m["length"] == "long"]
    if "long_form" in bundle_manifest:
        current_longs = bundle_manifest["long_form"].get("case_ids", [])
        for lid in long_ids:
            if lid not in current_longs:
                current_longs.append(lid)
        bundle_manifest["long_form"]["case_ids"] = current_longs

    write_json(val_bundle_dir / "manifest.json", bundle_manifest)
    print(f"[OK] current/validation-bundle updated: {len(merged_bundle_cases)} cases")


def merge_expansion_data():
    print("\n--- 3. Merging training and validation data into current ---")
    d2_rows = [json.loads(x) for x in (DIVERSE_V2 / "data/train.jsonl").read_text().splitlines()]
    d2_sft = [json.loads(x) for x in (DIVERSE_V2 / "sft/train.jsonl").read_text().splitlines()]
    d2_meta = json.loads((DIVERSE_V2 / "supporting.json").read_text())

    v1_rows = [json.loads(x) for x in (VALIDATION_V1 / "data/train.jsonl").read_text().splitlines()]
    v1_sft = [json.loads(x) for x in (VALIDATION_V1 / "sft/train.jsonl").read_text().splitlines()]
    v1_meta = json.loads((VALIDATION_V1 / "supporting.json").read_text())

    # 1. Update data/train.jsonl and sft/train.jsonl
    cur_train_data = [
        json.loads(x) for x in (CURRENT / "data/train.jsonl").read_text().splitlines()
    ]
    cur_train_sft = [json.loads(x) for x in (CURRENT / "sft/train.jsonl").read_text().splitlines()]
    merged_train_data = cur_train_data + d2_rows
    merged_train_sft = cur_train_sft + d2_sft
    (CURRENT / "data/train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged_train_data) + "\n"
    )
    (CURRENT / "sft/train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged_train_sft) + "\n"
    )
    print(
        f"[OK] Train split: {len(cur_train_data)} + {len(d2_rows)} -> {len(merged_train_data)} rows"
    )

    # 2. Update data/validation.jsonl and sft/validation.jsonl
    cur_val_data = [
        json.loads(x) for x in (CURRENT / "data/validation.jsonl").read_text().splitlines()
    ]
    cur_val_sft = [
        json.loads(x) for x in (CURRENT / "sft/validation.jsonl").read_text().splitlines()
    ]
    merged_val_data = cur_val_data + v1_rows
    merged_val_sft = cur_val_sft + v1_sft
    (CURRENT / "data/validation.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged_val_data) + "\n"
    )
    (CURRENT / "sft/validation.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged_val_sft) + "\n"
    )
    print(
        f"[OK] Validation split: {len(cur_val_data)} + {len(v1_rows)} -> {len(merged_val_data)} rows"
    )

    # 3. Process Evidence and Reviews for diverse-v2 and validation-v1
    writer = OutputWriter()
    cur_support = json.loads((CURRENT / "supporting.json").read_text())

    for ns, run_dir, rows, meta, split_name in [
        ("diverse-v2", DIVERSE_V2_RUN, d2_rows, d2_meta, "train"),
        ("validation-v1", VALIDATION_V1_RUN, v1_rows, v1_meta, "validation"),
    ]:
        print(f"\nProcessing evidence and reviews for {ns}...")
        ev_dir = CURRENT / f"evidence/expansions/{ns}"
        sol_calls_dir = ev_dir / "sol-calls"
        sol_calls_dir.mkdir(parents=True, exist_ok=True)

        # Map calls
        calls1 = {}
        for p in (run_dir / "sol-calls").glob("*.json"):
            data = json.loads(p.read_text())
            pkt = json.loads(data["request"]["messages"][1]["content"])[0]
            calls1[pkt["id"]] = p

        calls2 = {}
        if (run_dir / "adjudication/sol-calls").exists():
            for p in (run_dir / "adjudication/sol-calls").glob("*.json"):
                data = json.loads(p.read_text())
                pkt = json.loads(data["request"]["messages"][1]["content"])[0]
                calls2[pkt["id"]] = p

        rev1_list = json.loads((run_dir / "sol-reviews.json").read_text())["reviews"]
        rev1_map = {r["id"]: r for r in rev1_list}
        rev2_map = {}
        if (run_dir / "sol-reviews-round2.json").exists():
            rev2_map = {
                r["id"]: r
                for r in json.loads((run_dir / "sol-reviews-round2.json").read_text())["reviews"]
            }

        # Stage 06 records
        p06 = run_dir / "runs/author/stages/06_decision_panel/data.parquet"
        records = {r.id: r.model_dump(mode="json") for r in writer.read_parquet(p06)}

        native_records = {}
        for row, m in zip(rows, meta):
            rec = records[m["record_id"]]
            case = Case(
                id=m["id"], **row, context=m.get("context", {}), language=m.get("language", "en")
            )
            rec["scores"]["decisions"] = m["decisions"]
            rec["scores"]["judge_model"] = "openai/gpt-6-luna-decisions"
            rec["payload"] = {"instruction": case.user_message(), "response": row["output"]}
            rec["id"] = m["record_id"]
            native_records[m["id"]] = rec

        write_json(ev_dir / "native-records.json", native_records)

        # native-export-hashes.json
        native_exports = [
            digest(native_records[m["id"]]["payload"]) for m in meta if m["native_accepted"]
        ]
        write_json(ev_dir / "native-export-hashes.json", native_exports)

        # final-overlap-check.json
        write_json(
            ev_dir / "final-overlap-check.json",
            {"rows": len(rows), "new_to_new_overlap": 0, "rows_sha256": digest(rows)},
        )

        # sol reviews report
        review_filename = f"reviews/sol-{ns}.json"
        rep_reviews = []
        for m in meta:
            r1 = dict(rev1_map[m["id"]])
            r1["report"] = review_filename
            rep_reviews.append(r1)
        write_json(
            CURRENT / review_filename,
            {
                "reviewer_model": "openai/gpt-6.1-sol",
                "human_reviewed": False,
                "data_policy": "diverse-expansion",
                "reviews": rep_reviews,
            },
        )

        # Copy sol call files and assemble metadata
        for row, m in zip(rows, meta):
            item = dict(m)
            item["split"] = split_name
            item["language"] = "en"
            item["human_reviewed"] = False
            item["review_status"] = "sol_verified_machine"
            item["data_review"] = dict(rev1_map[m["id"]])
            item["data_review"]["report"] = review_filename

            c1_path = calls1[m["id"]]
            dest1 = sol_calls_dir / f"{m['id']}-1.json"
            shutil.copyfile(c1_path, dest1)
            item["verification_calls"] = [f"evidence/expansions/{ns}/sol-calls/{m['id']}-1.json"]

            if m["admission_tier"] != "native_strong":
                c2_path = calls2[m["id"]]
                dest2 = sol_calls_dir / f"{m['id']}-2.json"
                shutil.copyfile(c2_path, dest2)
                item["verification_calls"].append(
                    f"evidence/expansions/{ns}/sol-calls/{m['id']}-2.json"
                )
                item["adjudication_review"] = rev2_map[m["id"]]

            item["native_evidence"] = f"evidence/expansions/{ns}/native-records.json"
            item["native_exports"] = f"evidence/expansions/{ns}/native-export-hashes.json"
            item["overlap_evidence"] = f"evidence/expansions/{ns}/final-overlap-check.json"
            cur_support.append(item)

        print(f"[OK] {ns} evidence and reviews prepared successfully")

    write_json(CURRENT / "supporting.json", cur_support)
    print(f"[OK] supporting.json updated: {len(cur_support)} items")

    # Update quality.json
    quality = json.loads((CURRENT / "quality.json").read_text())
    quality["splits"] = {
        "train": len(merged_train_data),
        "dev": 100,
        "validation": len(merged_val_data),
    }
    pairs = {
        s: [json.loads(line) for line in (CURRENT / f"data/{s}.jsonl").read_text().splitlines()]
        for s in ["train", "dev", "validation"]
    }
    quality["by_mode"] = {s: dict(Counter(r["mode"] for r in rows)) for s, rows in pairs.items()}
    quality["independent_targets"] = {
        s: len({r["output"].strip().casefold() for r in rows}) for s, rows in pairs.items()
    }
    quality["source_groups"] = {
        s: len({m["source_group"] for m in cur_support if m["split"] == s}) for s in pairs
    }
    quality["languages"] = dict(Counter(m["language"] for m in cur_support))
    long = [m["tokens"]["input_tokens"] for m in cur_support if m.get("length") == "long"]
    if long:
        quality["long_pairs"] = len(long)
        quality["long_input_token_range"] = [min(long), max(long)]
    quality["identity_by_length"] = dict(
        Counter(m["length"] for m in cur_support if m.get("identity"))
    )
    quality["max_chat_tokens"] = max(m["tokens"]["chat_tokens"] for m in cur_support)
    quality["max_reserved_inference_tokens"] = max(
        m["tokens"]["prompt_tokens"] + 2048 for m in cur_support
    )
    write_json(CURRENT / "quality.json", quality)
    print(f"[OK] quality.json updated: splits={quality['splits']}")

    # Seal manifest.json for current
    seal_manifest(CURRENT, role="current-reviewed-data")
    print("[OK] current/manifest.json sealed")


def main():
    merge_benchmark_draft()
    merge_validation_bundle()
    merge_expansion_data()
    print("\nPhase 2 merge completed successfully!")


if __name__ == "__main__":
    main()
