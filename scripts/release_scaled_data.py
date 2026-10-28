"""Select exactly 1,000 train and 100 dev Opus passes, preserving v2 validation bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from finetuning_labs.rewriter.evaluation import write_json
from finetuning_labs.rewriter.reviewed_publication import require_script_review
from finetuning_labs.rewriter.reviewed_release import export_release, load_reviews, select_reviewed
from finetuning_labs.rewriter.scale_data import V2, select_balanced

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "reviewed-v3-1000"


def prepare(
    run: Path, destination: Path, script_review: Path | None, extra_runs=(), *, audit_only=False
):
    if destination.exists():
        raise FileExistsError(destination)
    script = ROOT / "src/finetuning_labs/rewriter/colab_training.py"
    notebook = ROOT / "notebooks/rewriter_train_l4.ipynb"
    cpu_report = ROOT / "artifacts/training-cpu-integration-v3/verification.json"
    if not audit_only:
        if script_review is None:
            raise ValueError("Publication requires the GPT-6.1 Sol implementation review")
        review = json.loads(script_review.read_text())
        require_script_review(review, script.read_bytes(), notebook.read_bytes())
        cpu = json.loads(cpu_report.read_text())
        if (
            cpu["script_sha256"] != hashlib.sha256(script.read_bytes()).hexdigest()
            or cpu["student_weights_loaded"]
            or cpu["optimizer_steps"] != 2
            or cpu["maximum_parameter_change"] <= 0
            or cpu["resumed_through_step"] != 2
        ):
            raise ValueError("Exact training script needs the CPU update and recovery verification")
    old_meta = json.loads((V2 / "supporting.json").read_text())
    pool = []
    for split in ["train", "dev", "validation"]:
        rows = [json.loads(x) for x in (V2 / f"data/{split}.jsonl").read_text().splitlines()]
        for row, meta in zip(rows, [m for m in old_meta if m["split"] == split], strict=True):
            pool.append((row, {**meta, "carried_from_v2": True}))
    reports = [V2 / name for name in sorted({m["opus_review"]["report"] for m in old_meta})]
    reviews = load_reviews(reports)
    new_reviews = []
    native_stats = []
    runs = [run, *extra_runs]
    shards = [
        (f"{i:02}-{p.name}", p)
        for i, source in enumerate(runs)
        for p in sorted((source / "review-shards").glob("*"))
    ]
    for _, shard in shards:
        if not (shard / "summary.json").exists():
            raise ValueError(f"Review shard incomplete: {shard.name}")
        data = shard / "generation"
        rows = [json.loads(x) for x in (data / "candidates.jsonl").read_text().splitlines()]
        metadata_path = shard / "candidate-supporting.json"
        if rows and not metadata_path.exists():
            raise ValueError("Missing mechanical and overlap review")
        metadata = json.loads(metadata_path.read_text()) if rows else []
        pool.extend(zip(rows, metadata, strict=True))
        report = shard / "opus-reviews.json"
        if report.exists():
            entries = json.loads(report.read_text())["reviews"]
            new_reviews.extend(entries)
            reviews.update(load_reviews([report]))
        stats = data / "native-checks/runs/checks/stages/02p_decision_panel/stats.json"
        if stats.exists():
            native_stats.append(json.loads(stats.read_text()))
    passed, excluded = select_reviewed(pool, reviews)
    chosen = select_balanced(passed)
    if len({r["output"] for r, m in chosen if m["split"] == "train"}) != 1000:
        raise ValueError("Training targets must be independent")
    audit = run / ("selection-audit-pool" if audit_only else "selected-pool")
    if audit.exists():
        raise FileExistsError(audit)
    audit.mkdir()
    (audit / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r, m in chosen)
    )
    write_json(audit / "scored-supporting.json", [m for r, m in chosen])
    aggregate = run / "opus-v3.json"
    write_json(
        aggregate,
        {
            "reviewer_model": "anthropic/claude-opus-5.5",
            "scope": "input, output and pair",
            "coverage_count": len(new_reviews),
            "reviews": new_reviews,
            "human_reviewed": False,
        },
    )
    quality = export_release(audit, [*reports, aggregate], destination)

    def copy(source, name):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())

    for report in reports:
        copy(report, "reviews/" + report.name)
    copy(aggregate, "reviews/opus-v3.json")
    if audit_only:
        for kind in ["data", "sft"]:
            if (destination / f"{kind}/validation.jsonl").read_bytes() != (
                V2 / f"{kind}/validation.jsonl"
            ).read_bytes():
                raise ValueError("Frozen v2 validation bytes changed")
        quality.update(
            status="local_data_selection_only",
            training_script_review="pending",
            excluded_rows=len(excluded),
            unused_passing_candidates=len(passed) - len(chosen),
        )
        write_json(destination / "quality.json", quality)
        write_json(destination / "excluded.json", excluded)
        files = {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(
            destination / "manifest.json", {"role": "local_data_selection_only", "files": files}
        )
        # No runtime, notebook, README or license card is emitted. This local
        # audit cannot pass audit_publication or be used as a Hub release.
        return quality
    copy(script, "runtime/train_rewriter.py")
    copy(notebook, "notebooks/rewriter_train_l4.ipynb")
    copy(script_review, "runtime/script-review.json")
    copy(
        ROOT / "artifacts/opus-training-script-review-round1.json",
        "runtime/review-history/opus-round1.json",
    )
    copy(cpu_report, "runtime/cpu-integration.json")
    for i, source in enumerate(runs):
        for name in ["source-plan.json", "pipeline.yaml", "rubric.yaml"]:
            copy(source / "generation" / name, f"evidence/generation/run-{i:02}/" + name)
        copy(
            source / "generation/runs/author/stages/02_generate/raw_responses.jsonl",
            f"evidence/generation/run-{i:02}/raw-responses.jsonl",
        )
    write_json(destination / "evidence/excluded-rows.json", excluded)
    write_json(destination / "excluded.json", excluded)
    write_json(destination / "evidence/native-decision-stats.json", native_stats)
    write_json(
        destination / "evidence/review-coverage.json",
        {
            "reviewed_exact_rows": len(new_reviews),
            "verdicts": dict(Counter(r["verdict"] for r in new_reviews)),
            "selected_new_train_and_dev": sum(not m.get("carried_from_v2") for r, m in chosen),
            "unused_passing_candidates": len(passed) - len(chosen),
        },
    )
    for label, shard in shards:
        for call in (shard / "opus-calls").glob("*.json"):
            copy(call, f"evidence/opus-calls/{label}-{call.name}")
        copy(shard / "completed-source-calls.json", f"evidence/source-shards/{label}.json")
    for source in (V2 / "validation-bundle").rglob("*"):
        if source.is_file():
            copy(source, "validation-bundle/" + str(source.relative_to(V2 / "validation-bundle")))
    for kind in ["data", "sft"]:
        if (destination / f"{kind}/validation.jsonl").read_bytes() != (
            V2 / f"{kind}/validation.jsonl"
        ).read_bytes():
            raise ValueError("Frozen v2 validation bytes changed")
    copy(V2 / "DATA_LICENSE.md", "DATA_LICENSE.md")
    quality.update(
        status="opus_verified_scaled_release",
        excluded_rows=len(excluded),
        unused_passing_candidates=len(passed) - len(chosen),
        frozen_validation_version="reviewed-v2",
        independent_human_calibration=False,
        student_training_run=False,
    )
    write_json(destination / "quality.json", quality)
    table = "\n".join(
        f"| {mode} | 250 | 25 | {quality['by_mode']['validation'][mode]} |"
        for mode in quality["by_mode"]["train"]
    )
    configs = "\n".join(
        f"- config_name: {name}\n  data_files:\n"
        + "\n".join(
            f"  - split: {split}\n    path: {PREFIX}/{kind}/{split}.jsonl"
            for split in ["train", "dev", "validation"]
        )
        for name, kind in [("default", "data"), ("sft", "sft")]
    )
    (destination / "README.md").write_text(f"""---
language: [en, de, pt]
license: other
license_name: private-reviewed-rewriter-mixed-sources
license_link: https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/blob/main/{PREFIX}/DATA_LICENSE.md
task_categories: [text-generation]
configs:
{configs}
---
# Reviewed rewriter training data, 1,000 examples

Use `{PREFIX}/`. Training has exactly 1,000 pairs, development has 100, and validation has 84.
The previous v2 validation texts and order are unchanged. Old releases remain available;
the root 219-pair pilot is retired. There are 250 training and 25 development pairs per mode.

| Mode | Train | Dev | Validation |
| --- | ---: | ---: | ---: |
{table}

Luna authors new fictional episodes through the pinned native Arka pipeline. Luna Decisions
checks the candidate input, output quality, facts, intent, literals and mode. Independent Opus
verifies each retained exact input, output and pair. Uncertain, failed, missing, malformed,
oversized and overlapping candidates are excluded. Reused v2 pairs retain their exact reviews.
Opus did not author or edit data. GPT-6.1 Sol reviews the training script and notebook;
its exact-file implementation report is in `runtime/script-review.json`.

Data rows have only input, output, mode and checks. SFT rows have only system, user and assistant.
Context needed for rewriting enters the user request; all other metadata stays outside loss.
All chats fit 4,096 tokens without truncation and reserve 2,048 tokens for generation. Long
examples, quotations, numbered sections and unchanged examples remain included. See quality.json
for measured coverage. Source groups do not cross splits. Training has 1,000 distinct targets.

The notebook in `notebooks/rewriter_train_l4.ipynb` loads the private Hub files and pinned
LiquidAI/LFM2.5-230M weights. Use an L4 runtime and a read-only HF_TOKEN in Colab Secrets.
It checks data, saves stock outputs, runs an optional memorization proof, trains on train with
checkpoint selection from dev, and saves tuned outputs on the frozen validation set.
It uses Transformers full fine-tuning and assistant-only labels through EOS. Checkpoints and
reports go to Drive by default. Checkpoints rotate on local disk; two fixed verified recovery
archives persist on Drive. The proof saves metrics only. Have 10 GiB free on Drive and 12 GiB
on local disk before training. Completed stages skip after verifying their artifact hashes.
It does not publish weights automatically. The actual 230M
student has not been trained as part of this data preparation.

These are model judgments, not human gold. Synthetic dictation is not recorded ASR evidence.
Real user preferences, independent human calibration and an unseen final test remain pending.
Preserve TypeWhisper's GPL v3 source snapshot and notice in validation-bundle/supporting/.
No personal transcriptions, recordings or credentials are included. All publication is private.
""")
    files = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file() and p != destination / "manifest.json"
    }
    write_json(destination / "manifest.json", {"role": PREFIX, "files": files})
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script-review", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--extra-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.run,
                args.output,
                args.script_review,
                args.extra_run,
                audit_only=args.audit_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
