"""Export the reviewed v2 data and explicit evidence without replacing the original benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from arka.pipeline.output import OutputWriter

from finetuning_labs.rewriter.arka_pipeline import ARKA_PIN
from finetuning_labs.rewriter.benchmark import Case, load_bundle
from finetuning_labs.rewriter.evaluation import digest
from finetuning_labs.rewriter.reviewed_release import export_release

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    "round2-train-dev.json",
    "round2-validation.json",
    "round3-train-dev-recovered.json",
    "round3-validation.json",
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def card(quality):
    table = "\n".join(
        f"| {mode} | "
        + " | ".join(
            str(quality["by_mode"][s].get(mode, 0)) for s in ["train", "dev", "validation"]
        )
        + " |"
        for mode in ["/slack", "/transcribe", "/prompt", "/polish"]
    )
    return f"""---
language:
- en
- de
- pt
license: other
license_name: private-reviewed-rewriter-mixed-sources
license_link: https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/blob/main/reviewed-v2/DATA_LICENSE.md
task_categories:
- text-generation
tags:
- text-rewriting
- synthetic
- liquid
configs:
- config_name: default
  data_files:
  - split: train
    path: reviewed-v2/data/train.jsonl
  - split: dev
    path: reviewed-v2/data/dev.jsonl
  - split: validation
    path: reviewed-v2/data/validation.jsonl
- config_name: sft
  data_files:
  - split: train
    path: reviewed-v2/sft/train.jsonl
  - split: dev
    path: reviewed-v2/sft/dev.jsonl
  - split: validation
    path: reviewed-v2/sft/validation.jsonl
---
# Reviewed rewriter data v2

Use `reviewed-v2/`. The original 219-pair pilot in the repository root failed independent
Opus data verification and is retired for training. Its old files remain for reproduction.

This version has 65 training, 17 development and 84 validation pairs for
`LiquidAI/LFM2.5-230M`. Each released input, output and pair passed an exact-row Opus 5.5
review. Two unresolved rows are excluded. These are model judgments, not human gold labels.

| Mode | Train | Dev | Validation |
| --- | ---: | ---: | ---: |
{table}

## Sources and formats

GPT-6 Luna generated one fictional pair per independent scenario through clean Arka commit
`{ARKA_PIN}`. The native pipeline ran generation, normalization, exact and length checks,
`decision_panel`, and `labeling_engine` with Luna Decisions. The parent corrected defects from
data-only Opus reviews and resubmitted every changed row. Opus did not generate or edit data.

The 84 validation pairs contain all 68 original benchmark cases, with two versioned corrections,
plus 16 new pairs. Thirty original cases derive from TypeWhisper authored fixtures. The rest
are authored or synthetic. No personal transcriptions or recordings are included. Preserve the
TypeWhisper GPL v3 notice and source snapshot supplied in `validation-bundle/supporting/`.
The original 68-case files and their historical model scores remain unchanged elsewhere.

Default rows have only `input`, `output`, `mode` and `checks`. SFT rows have only `system`,
`user` and `assistant`, as in the supplied formatter guide. Scores, critiques, IDs and source
records are separate. Only required language, profile and dictionary context enters the request.

## Quality evidence

All 168 candidate pairs were tested with six native decision dimensions. Arka automatically
accepted 75. The independent Opus reviews adjudicate the other candidates; native scores and
policy outcomes remain visible in `supporting.json`. A released row does not imply a native
automatic pass. Probabilities have not been checked against independent human labels.

The 166 retained pairs include 33 long inputs of 1,094 to 1,906 pinned Liquid tokens. Unchanged
examples occur at short, medium and long lengths. Training has 65 distinct source groups and
65 distinct targets. No source group crosses any of the three splits. Exact, embedded and
near-copy checks found no overlap between training and the original benchmark, its fixtures,
or the local personal holdout. Wording checks do not prove semantic independence.

Every complete chat fits 4,096 tokens, and every prompt leaves room for 2,048 generated tokens.
The maximum complete chat is {quality["max_chat_tokens"]} tokens. No example was truncated.
Train on assistant content through EOS only. Mask system/user tokens and the template's trailing
newline with -100. The Colab dataset-check notebook verifies these spans without loading weights.

Use `dev` for checkpoint selection and `validation` for the fixed stock-versus-tuned comparison.
The latter has 80 source groups and 80 distinct targets because some original long cases are
related mode variants. This is development validation, not an unseen final test. Rerun the stock
baseline on v2 before comparing it with a fine-tune; historical v1 scores are not v2 scores.

The dataset is small and predominantly English. Inputs are synthetic drafts and dictation,
not measured ASR output from recordings. Dictionary and profile coverage remain limited.
Uncertainty-preservation cases are overrepresented. No student training or quality improvement
is claimed. Human calibration and broader real-use evaluation remain pending.

`quality.json` records counts and limits. `reviews/` contains exact verdicts and the report-recovery
evidence. `evidence/` retains source plans, generation attempts, corrections, rejected candidates,
native records and pipeline statistics. These evidence files are not training inputs.
"""


def prepare(pool: Path, destination: Path):
    review_root = ROOT / "artifacts/opus-reviews"
    review_packets = {
        "round2-train-dev.json": [
            ROOT / "artifacts/arka-native-v2/opus-train",
            ROOT / "artifacts/arka-native-v2/opus-dev",
        ],
        "round2-validation.json": [
            ROOT / "artifacts/arka-validation-v2/opus-inputs",
            ROOT / "artifacts/arka-native-v2/opus-validation",
        ],
        "round3-train-dev-recovered.json": [pool / "opus-pending-train-dev"],
        "round3-validation.json": [pool / "opus-pending-validation"],
    }
    coverage = {}
    for name, folders in review_packets.items():
        expected = [
            (r["id"], r["row_sha256"])
            for folder in folders
            for path in folder.glob("*.jsonl")
            for r in [json.loads(x) for x in path.read_text().splitlines()]
        ]
        report = json.loads((review_root / name).read_text())
        actual = [(r["id"], r["row_sha256"]) for r in report["reviews"]]
        assert len(actual) == len(set(actual)) == report["coverage_count"]
        assert sorted(actual) == sorted(expected), name
        coverage[name] = {
            "expected": len(expected),
            "verified": len(actual),
            "all_exact_ids_and_hashes_match": True,
        }
    quality = export_release(pool, [review_root / x for x in REPORTS], destination)
    write_json(destination / "evidence/review-coverage.json", coverage)
    for name, folders in review_packets.items():
        packet = [
            json.loads(line)
            for folder in folders
            for path in folder.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        path = destination / "evidence/review-packets" / (Path(name).stem + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packet))

    def copy(source, name):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())

    for name in REPORTS + ["round3-train-dev.json", "round3-train-dev-command.txt"]:
        copy(review_root / name, "reviews/" + name)
    copy(
        ROOT / "notebooks/rewriter_reviewed_data_check.ipynb",
        "notebooks/rewriter_reviewed_data_check.ipynb",
    )
    for name in [
        "candidates.jsonl",
        "supporting.json",
        "scored-supporting.json",
        "corrections.json",
    ]:
        copy(pool / name, "evidence/pool/" + name)
    for name in [
        "pipeline.yaml",
        "rubric.yaml",
        "seeds.jsonl",
        "native-output.jsonl",
        "runs/checks/report/run_report.json",
        "runs/checks/stages/02p_decision_panel/stats.json",
        "runs/checks/stages/03_label_quality/stats.json",
    ]:
        copy(pool / "native-checks" / name, "evidence/native/" + name)
    writer = OutputWriter()
    records = writer.read_parquet(
        pool / "native-checks/runs/checks/stages/02p_decision_panel/data.parquet"
    )
    path = destination / "evidence/native/decision-records.jsonl"
    path.write_text(
        "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in records)
    )
    for run in [
        "arka-native-v2",
        "arka-native-v2-length-repairs",
        "arka-native-v2-extra-identities",
    ]:
        source = ROOT / "artifacts" / run
        for name in [
            "source-plan.json",
            "pipeline.yaml",
            "rubric.yaml",
            "seeds.jsonl",
            "rejected.json",
            "candidates.jsonl",
            "supporting.json",
        ]:
            copy(source / name, f"evidence/generation/{run}/{name}")
        generated = writer.read_parquet(source / "runs/author/stages/02_generate/data.parquet")
        path = destination / f"evidence/generation/{run}/generated-records.jsonl"
        path.write_text(
            "".join(
                json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in generated
            )
        )

    native = {
        digest(x)
        for x in [
            json.loads(line)
            for line in (pool / "native-checks/native-output.jsonl").read_text().splitlines()
        ]
    }
    support = json.loads((destination / "supporting.json").read_text())
    original = json.loads(
        (ROOT / "data/rewriter/benchmark-draft/supporting/cases.json").read_text()
    )
    old_meta = {m["id"]: m for m in original}
    rows_by_split = {
        s: [json.loads(x) for x in (destination / f"data/{s}.jsonl").read_text().splitlines()]
        for s in ["train", "dev", "validation"]
    }
    for split, rows in rows_by_split.items():
        for row, meta in zip(rows, [m for m in support if m["split"] == split], strict=True):
            case = Case(id=meta["id"], **row, language=meta["language"], context=meta["context"])
            candidate = {
                "instruction": case.user_message(),
                "response": case.output,
                "system": None,
                "turns": None,
            }
            meta["native_policy_accepted"] = digest(candidate) in native
            meta["release_adjudication"] = "independent_exact_row_opus_pass"
            meta["source_kind"] = (
                old_meta[meta["id"]]["source_kind"]
                if meta["id"] in old_meta
                else "synthetic_arka_luna"
            )
    write_json(destination / "supporting.json", support)
    quality["native_policy"] = {
        "candidates_scored": 168,
        "automatically_accepted": 75,
        "released_automatically_accepted": sum(m["native_policy_accepted"] for m in support),
        "human_calibrated": False,
    }
    quality["source_composition"] = {
        "synthetic_new_pairs": 98,
        "original_benchmark_pairs": 68,
        "original_references_corrected_in_v2": 2,
    }
    write_json(destination / "quality.json", quality)

    bundle = destination / "validation-bundle"
    copy(destination / "data/validation.jsonl", "validation-bundle/cases.jsonl")
    validation = [m for m in support if m["split"] == "validation"]
    write_json(bundle / "supporting/cases.json", validation)
    (bundle / "supporting/fixtures.jsonl").write_text("")
    old = ROOT / "data/rewriter/benchmark-draft"
    for name in [
        "supporting/judge-system.txt",
        "supporting/TYPEWHISPER-LICENSE.txt",
        "supporting/typewhisper-cases.jsonl",
    ]:
        copy(old / name, "validation-bundle/" + name)
    original_manifest = json.loads((old / "manifest.json").read_text())
    bundle_files = {
        str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in bundle.rglob("*")
        if p.is_file()
    }
    write_json(
        bundle / "manifest.json",
        {
            "schema_version": "rewriter-reviewed-validation-v2",
            "sources": original_manifest["sources"],
            "status": "machine_verified_development_validation",
            "files": bundle_files,
            "original_benchmark_modified": False,
            "locked_final_test": False,
            "long_form": {"case_ids": [m["id"] for m in validation if m["length"] == "long"]},
        },
    )
    assert len(load_bundle(bundle).cases) == 84

    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / "scripts/curate_arka_v2.py",
            ROOT / "scripts/recover_opus_report.py",
            Path(__file__),
            ROOT / "src/finetuning_labs/rewriter/source_briefs.py",
            ROOT / "src/finetuning_labs/rewriter/arka_pipeline.py",
            ROOT / "src/finetuning_labs/rewriter/reviewed_release.py",
            ROOT / "src/finetuning_labs/rewriter/reviewed_publication.py",
            ROOT / "notebooks/rewriter_reviewed_data_check.ipynb",
        ]
    }
    write_json(
        destination / "provenance.json",
        {
            "arka_revision": ARKA_PIN,
            "teacher": "openai/gpt-6-luna",
            "decision_model": "openai/gpt-6-luna-decisions",
            "verifier": "Opus 5.5",
            "verifier_role": "data verification only",
            "human_labels": False,
            "source_code_sha256": source_hashes,
            "original_benchmark_manifest_sha256": hashlib.sha256(
                (old / "manifest.json").read_bytes()
            ).hexdigest(),
            "report_recovery": "reviews/round3-train-dev-command.txt preserves the reviewer's omitted array literals.",
        },
    )
    (destination / "README.md").write_text(card(quality))
    (destination / "DATA_LICENSE.md").write_text(
        "# Data provenance and source notices\n\n"
        "This private release combines new fictional synthetic pairs and the original authored "
        "benchmark. No personal transcriptions or recordings are included. No public redistribution "
        "license has been assigned to the new authored or synthetic material.\n\n"
        "Thirty validation cases derive from TypeWhisper authored fixtures at commit "
        "c878eced59fc4ccc7755ccc5f30b1fbcc62c1939. The repository supplies GPL v3 text, and no "
        "separate fixture grant was verified. Preserve the complete original source snapshot and "
        "TYPEWHISPER-LICENSE.txt in validation-bundle/supporting/. This release does not relabel "
        "those fixtures or their derivatives under another license. Training and dev contain "
        "new synthetic scenarios, not those fixtures.\n\n"
        "Source repository: https://github.com/jayshah5696/typewhisper-linux\n"
    )
    hashes = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file() and p != destination / "manifest.json"
    }
    write_json(destination / "manifest.json", {"role": "reviewed_rewriter_v2", "files": hashes})
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pool, args.output), indent=2))


if __name__ == "__main__":
    main()
