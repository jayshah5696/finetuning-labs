"""Publish only Parquet datasets and source notices to the Hugging Face data repositories."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

PAIR_COLUMNS = ("input", "output", "mode", "checks")
SFT_COLUMNS = ("system", "user", "assistant")
SPLITS = ("train", "dev", "validation")
PROJECT = Path(__file__).resolve().parents[3]
TRAIN_REPO = "jayshah5696/finetuning-labs-rewriter-training"
BENCHMARK_REPO = "jayshah5696/finetuning-labs-rewriter"


def english_splits(rows, guides, support):
    selected_rows, selected_guides = {}, {}
    for split in rows:
        metadata = [m for m in support if m["split"] == split]
        selected_rows[split], selected_guides[split] = [], []
        for row, guide, meta in zip(rows[split], guides[split], metadata, strict=True):
            if guide["user"].splitlines()[1] != "LANGUAGE: " + meta["language"]:
                raise ValueError("Verified language and SFT header differ")
            if meta["language"] == "en":
                selected_rows[split].append(row)
                selected_guides[split].append(guide)
    return selected_rows, selected_guides


def _parquet(rows, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows or any(
        set(r) != set(columns) or any(not isinstance(r[k], str) for k in columns) for r in rows
    ):
        raise ValueError("Data columns must be the documented string fields only")
    table = pa.Table.from_pylist(rows, schema=pa.schema([(k, pa.string()) for k in columns]))
    target = pa.BufferOutputStream()
    pq.write_table(table, target, compression="zstd")
    data = target.getvalue().to_pybytes()
    if pq.read_table(io.BytesIO(data)).to_pylist() != rows:
        raise ValueError("Parquet roundtrip changed data")
    return data


def build_dataset(
    rows, guides, *, title, description, license_text, source_license, languages=None
):
    import yaml

    if set(rows) != set(guides) or not set(rows).issubset(SPLITS):
        raise ValueError("Dataset split names differ")
    files, configs = {}, []
    for split in rows:
        if len(rows[split]) != len(guides[split]):
            raise ValueError("Data and SFT counts differ")
        for row, guide in zip(rows[split], guides[split], strict=True):
            if (
                guide.get("assistant") != row.get("output")
                or not guide.get("user", "").endswith("INPUT:\n" + row["input"])
                or not guide["user"].startswith(f"MODE: {row['mode']}\n")
            ):
                raise ValueError("SFT input, context or target differs from the readable data")
        files[f"data/{split}.parquet"] = _parquet(rows[split], PAIR_COLUMNS)
        files[f"sft/{split}.parquet"] = _parquet(guides[split], SFT_COLUMNS)
    for config, kind in [("default", "data"), ("sft", "sft")]:
        item = {
            "config_name": config,
            "data_files": [{"split": s, "path": f"{kind}/{s}.parquet"} for s in rows],
        }
        if config == "default":
            item["default"] = True
        configs.append(item)
    metadata = {
        "pretty_name": title,
        "language": languages or ["en", "de", "pt"],
        "license": "other",
        "task_categories": ["text-generation"],
        "tags": ["text-rewriting", "synthetic"],
        "configs": configs,
    }
    table = "\n".join(f"| {s} | {len(v)} |" for s, v in rows.items())
    card = f"""---
{yaml.safe_dump(metadata, sort_keys=False).strip()}
---

# {title}

{description}

| Split | Rows |
| --- | ---: |
{table}

The default viewer shows input, output, mode and checks. Select `sft` to see system, user and
assistant. The SFT user text includes the required language, dictionary and profile context.
Both configurations contain the same examples in the same order.

```python
from datasets import load_dataset

dataset = load_dataset("{"jayshah5696/finetuning-labs-rewriter-training" if "train" in rows else "jayshah5696/finetuning-labs-rewriter"}", "sft")
```

Data files are Parquet. This repository contains data and source notices only. Training code,
notebooks and verification records are in [Finetuning Labs](https://github.com/jayshah5696/jadoo-labs/tree/main/projects/finetuning-labs).
Review judgments are model judgments. Independent human labels and an unseen final test remain
pending. Synthetic dictation is not recorded speech-recognition evidence. No personal history
or recordings are included. See DATA_LICENSE.md for source rights and attribution.
"""
    files["README.md"] = card.encode()
    files["DATA_LICENSE.md"] = license_text
    files["licenses/TYPEWHISPER-LICENSE.txt"] = source_license
    return files


def cleanup_paths(existing, files):
    allowed = {"README.md", "DATA_LICENSE.md", "licenses/TYPEWHISPER-LICENSE.txt"}
    allowed |= {f"{kind}/{split}.parquet" for kind in ["data", "sft"] for split in SPLITS}
    if not set(files).issubset(allowed) or "README.md" not in files:
        raise ValueError("Hugging Face payload must contain data and source notices only")
    return sorted(set(existing) - set(files) - {".gitattributes"})


def _sources():
    from .benchmark import COMMON_SYSTEM, MODES, load_bundle
    from .reviewed_publication import audit_publication

    release = PROJECT / "data/rewriter/current"
    # Audit the retained Git records, then export only the data columns.
    audit_publication(release)
    from .boost_data import verify_holdouts

    freeze = json.loads((PROJECT / "data/rewriter/holdout-freeze.json").read_text())
    verify_holdouts({name: (release / name).read_bytes() for name in freeze}, freeze)
    rows = {
        s: [json.loads(x) for x in (release / f"data/{s}.jsonl").read_text().splitlines()]
        for s in SPLITS
    }
    guides = {
        s: [json.loads(x) for x in (release / f"sft/{s}.jsonl").read_text().splitlines()]
        for s in SPLITS
    }
    counts = {s: len(v) for s, v in rows.items()}
    if counts["train"] < 600 or counts["dev"] != 100 or counts["validation"] != 144:
        raise ValueError("Keep train rows and the frozen 100/144 holdouts")
    support = json.loads((release / "supporting.json").read_text())
    rows, guides = english_splits(rows, guides, support)
    from .evaluation import digest

    freeze = json.loads((PROJECT / "data/rewriter/english-holdout-freeze.json").read_text())
    for split, expected in freeze.items():
        ids = [m["id"] for m in support if m["split"] == split and m["language"] == "en"]
        if (
            len(rows[split]) != expected["count"]
            or digest(rows[split]) != expected["data_sha256"]
            or digest(guides[split]) != expected["sft_sha256"]
            or ids != expected["case_ids"]
        ):
            raise ValueError("Frozen English holdout view changed")
    counts = {s: len(v) for s, v in rows.items()}
    bundle = load_bundle()
    old_rows = [{k: getattr(case, k) for k in PAIR_COLUMNS} for case in bundle.cases]
    old_guides = [
        {"system": COMMON_SYSTEM + MODES[c.mode], "user": c.user_message(), "assistant": c.output}
        for c in bundle.cases
    ]
    license_text = (PROJECT / "data/rewriter/PUBLIC_DATA_LICENSE.md").read_bytes()
    source_license = (release / "validation-bundle/supporting/TYPEWHISPER-LICENSE.txt").read_bytes()
    return (
        [
            (
                TRAIN_REPO,
                rows,
                guides,
                "English rewriter training data",
                (
                    f"English only: {counts['train']:,} training pairs, {counts['dev']} development pairs and {counts['validation']} frozen development-validation pairs. "
                    "Expanded coverage includes 5 dictation families: developer coding symbols, workspace file context (@file tags), "
                    "spoken list structures, multi-level contrasts, and chained speech repairs / anti-chatbot isolation, "
                    "with comprehensive 500–2,000 token coverage across short, medium, and long contexts. "
                    "The English holdout cases retain their original texts and order. The full multilingual source holdouts remain in Git and historical Hub commits. "
                    "Use dev for checkpoint selection and "
                    "validation for the later comparison. All selected pairs retain exact-row model reviews."
                ),
            ),
            (
                BENCHMARK_REPO,
                {"validation": old_rows},
                {"validation": old_guides},
                "Rewriter calibration benchmark",
                f"The expanded {len(old_rows)} authored calibration cases. "
                "These historical references and expansion cases are preserved. The current training dataset has the "
                "separate frozen English validation view: "
                "[rewriter-training](https://huggingface.co/datasets/" + TRAIN_REPO + ").",
            ),
        ],
        license_text,
        source_license,
    )


def main():
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/public-parquet")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    sources, license_text, source_license = _sources()
    api, receipts = HfApi(), []
    for repo, rows, guides, title, description in sources:
        # Check uncompressed text, since a compressed Parquet byte scan is insufficient.
        text = json.dumps([rows, guides], ensure_ascii=False).encode()
        needles = []
        personal = PROJECT / "data/rewriter/personal/calibration-10"
        if personal.exists():
            needles += [
                json.loads(line)["input"].encode()
                for line in (personal / "cases.jsonl").read_text().splitlines()
            ]
            needles += [
                x["source_id"].encode()
                for x in json.loads((personal / "supporting/cases.json").read_text())
            ]
        if re.search(rb"sk-or-v1-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}", text) or any(
            n in text for n in needles
        ):
            raise ValueError("Private content in uncompressed data")
        files = build_dataset(
            rows,
            guides,
            title=title,
            description=description,
            license_text=license_text,
            source_license=source_license,
            languages=["en"] if repo == TRAIN_REPO else None,
        )
        output = args.output / repo.split("/")[1]
        output.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        cleanup_paths([], files)
        if not args.publish:
            print(
                json.dumps(
                    {
                        "repo": repo,
                        "files": sorted(files),
                        "splits": {s: len(r) for s, r in rows.items()},
                    }
                )
            )
            continue
        before = api.repo_info(repo, repo_type="dataset")
        existing = api.list_repo_files(repo, repo_type="dataset", revision=before.sha)
        removed = cleanup_paths(existing, files)
        commit = api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            parent_commit=before.sha,
            commit_message="Use public Parquet splits; keep code and audit artifacts in Git",
            operations=[
                *[CommitOperationDelete(path_in_repo=n) for n in removed],
                *[
                    CommitOperationAdd(path_in_repo=n, path_or_fileobj=data)
                    for n, data in files.items()
                ],
            ],
        )
        for name, data in files.items():
            downloaded = hf_hub_download(repo, name, repo_type="dataset", revision=commit.oid)
            if Path(downloaded).read_bytes() != data:
                raise ValueError(f"Uploaded bytes differ: {name}")
        actual = api.list_repo_files(repo, repo_type="dataset", revision=commit.oid)
        if set(actual) != set(files) | ({".gitattributes"} & set(existing)):
            raise ValueError("Non-data files remain in the current Hub tree")
        api.update_repo_settings(repo_id=repo, repo_type="dataset", private=False)
        if api.repo_info(repo, repo_type="dataset").private:
            raise ValueError("Dataset must be public for the viewer")
        receipt = {
            "repo_id": repo,
            "revision": commit.oid,
            "previous_revision": before.sha,
            "private": False,
            "removed_current_paths": removed,
            "files": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
            "splits": {s: len(r) for s, r in rows.items()},
        }
        receipts.append(receipt)
        print(
            json.dumps(
                {k: v for k, v in receipt.items() if k not in {"files", "removed_current_paths"}}
            ),
            flush=True,
        )
    if receipts:
        destination = PROJECT / "data/rewriter/releases/public-parquet.json"
        destination.write_text(json.dumps({"repositories": receipts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
