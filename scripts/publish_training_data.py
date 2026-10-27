"""Publish only a verified synthetic training release to a separate private dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

from finetuning_labs.rewriter.training import audit_run

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "jayshah5696/finetuning-labs-rewriter-training"


def dataset_card(repo, quality, summary):
    table = "\n".join(
        f"| {mode} | {quality['by_mode']['train'].get(mode, 0)} | "
        f"{quality['by_mode']['validation'].get(mode, 0)} |"
        for mode in ["/slack", "/transcribe", "/prompt", "/polish"]
    )
    return f"""---
language:
- en
- de
- pt
license: other
license_name: private-synthetic-pilot
license_link: https://huggingface.co/datasets/{repo}/blob/main/DATA_LICENSE.md
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
    path: train.jsonl
  - split: validation
    path: validation.jsonl
- config_name: sft
  data_files:
  - split: train
    path: sft/train.jsonl
  - split: validation
    path: sft/validation.jsonl
---
# Finetuning Labs rewriter training pilot

This private synthetic pilot targets `LiquidAI/LFM2.5-230M`. GPT-6 Luna authored new fictional
documents and input variants. GPT-6.1 Sol reviewed each exported pair with the same five-criterion
rubric used in the stock Liquid and Luna diagnostic comparison. The judge is not human calibrated.
These are machine-reviewed training targets, not human gold answers.

{quality["splits"].get("train", 0)} training pairs and
{quality["splits"].get("validation", 0)} internal validation pairs passed every release check.
The run scheduled {summary["planned_pairs"]} pairs from {summary["planned_documents"]} source
documents. Rejected, uncertain, invalid, and incomplete candidates are excluded. Related modes
and variants stay in one topic group. The release has {quality["source_documents"]} source
documents in {quality["source_groups"]} topic groups; variants are not independent documents.

| Mode | Train | Internal validation |
| --- | ---: | ---: |
{table}

## Simple formats

The default data has only `input`, `output`, `mode`, and `checks`. The `sft` config has only
`system`, `user`, and `assistant`, following the original local formatter guide. It uses the
same system and user prompt contract as the baseline. Do not feed `checks`, review labels,
source records, or judge critiques to the student. Supporting records stay outside supervised text.

Render the three SFT fields as one system/user/assistant conversation with the pinned Liquid
chat template. Train on assistant content through EOS only. Mask system/user tokens with -100.
The native template's newline after EOS is also masked. The preparation audit checked these spans.
No example was truncated. Use 4,096 tokens and retain the 2,048-token generation limit for evaluation.
Maximum complete training chat is {quality["max_chat_tokens"]} tokens.

## Quality checks and boundaries

Each accepted pair preserves its protected literals, passes meaning, terms and literals, mode,
readability, and output discipline, and has no reported critical error. Uncertain is not pass.
Long inputs contain 1,000 to 2,000 pinned Liquid tokens. The data includes paragraphs, numbered
sections, and direct quotes. Identity examples teach the student to leave correct text unchanged.

The user-accepted 68-case development-validation set remains in the separate
[validation repository](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter).
Its cases, references, output fixtures, and related source families are excluded here. Exact,
embedded, and close wording overlap checks found no overlap. Those checks do not prove that all
possible semantic similarities are absent. The private personal-transcription bundle is excluded.

The internal validation split uses different topic groups and is for checkpoint selection.
The accepted 68 cases remain the fixed before/after comparison. Neither set is an unseen final
test. All per-output independent human labels remain pending. This pilot is too small to establish
production quality. It supports the next Colab L4 training proof and a measured first fine-tune.

`quality.json` has coverage and token counts. `supporting.json` binds rows to source groups,
literal rules, language/context, and judge critiques. `audit/` retains exact source/judge calls,
the rejected-candidate ledger, and the verification report. The quarantined transport rehearsal
is excluded. No training run or model improvement is claimed by this dataset publication.
"""


def package(root: Path, repo: str) -> dict[str, bytes]:
    if root.resolve().is_relative_to((PROJECT / "data/rewriter/personal").resolve()):
        raise ValueError("personal data cannot be published")
    audit_run(root)
    spec = json.loads((root / "run.json").read_text())
    release = root / "release"
    quality = json.loads((release / "quality.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    names = [
        "train.jsonl",
        "validation.jsonl",
        "sft/train.jsonl",
        "sft/validation.jsonl",
        "supporting.json",
        "quality.json",
        "manifest.json",
    ]
    contents = {name: (release / name).read_bytes() for name in names}
    for name in ["run.json", "ledger.json", "summary.json", "audit.json"]:
        contents[f"audit/{name}"] = (root / name).read_bytes()
    for plan in spec["plans"]:
        name = f"sources/{plan['id']}.json"
        contents[f"audit/{name}"] = (root / name).read_bytes()
        for variant in ["identity", "light", "mode-noise", "challenge"]:
            name = f"judges/{plan['id']}-{variant}.json"
            if (root / name).exists():
                contents[f"audit/{name}"] = (root / name).read_bytes()
    contents["README.md"] = dataset_card(repo, quality, summary).encode()
    contents["DATA_LICENSE.md"] = (
        b"# Data provenance\n\nNew fictional synthetic documents and variants authored with "
        b"OpenAI GPT-6 Luna through OpenRouter. OpenAI GPT-6.1 Sol supplied automatic quality "
        b"reviews. No TypeWhisper fixtures, personal transcripts, or external text corpus is "
        b"included. This release is private. No public redistribution license has been assigned.\n"
    )
    needles = [
        value.encode() for value in [os.environ.get("OPENROUTER_API_KEY"), get_token()] if value
    ]
    personal = PROJECT / "data/rewriter/personal/calibration-10"
    if personal.exists():
        for line in (personal / "cases.jsonl").read_text().splitlines():
            needles.append(json.loads(line)["input"].encode())
        for item in json.loads((personal / "supporting/cases.json").read_text()):
            needles.append(item["source_id"].encode())
    for name, data in contents.items():
        if any(needle and needle in data for needle in needles):
            raise ValueError(f"private-content scan failed: {name}")
    return contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    contents = package(args.run, args.repo)
    (args.run / "release/README.md").write_bytes(contents["README.md"])
    (args.run / "release/DATA_LICENSE.md").write_bytes(contents["DATA_LICENSE.md"])
    if args.prepare_only:
        print(json.dumps({"prepared_files": len(contents), "repo": args.repo}))
        return
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="dataset")
    if not info.private:
        raise ValueError("Refusing to publish this pilot to a public repository")
    existing = set(api.list_repo_files(args.repo, repo_type="dataset")) - {".gitattributes"}
    if existing:
        raise ValueError("Destination already contains data; use a new versioned release")
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        parent_commit=info.sha,
        commit_message="Publish verified synthetic rewriter training pilot",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
            for name, data in contents.items()
        ],
    )

    def verify(item):
        name, data = item
        path = hf_hub_download(
            args.repo, name, repo_type="dataset", revision=commit.oid, force_download=True
        )
        if Path(path).read_bytes() != data:
            raise ValueError(f"Uploaded bytes differ: {name}")
        return name, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(verify, contents.items()))
    receipt = {
        "repo_id": args.repo,
        "revision": commit.oid,
        "private": True,
        "verified_file_count": len(hashes),
        "files": hashes,
        "url": f"https://huggingface.co/datasets/{args.repo}",
    }
    (PROJECT / "data/rewriter/releases/training-data.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
