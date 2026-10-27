"""Audit and publish an additive reviewed-data version to the private Hub repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .arka_pipeline import DIMENSIONS
from .benchmark import COMMON_SYSTEM, MODES, Case
from .evaluation import digest
from .reviewed_release import load_reviews

PREFIX = "reviewed-v2"
REPO = "jayshah5696/finetuning-labs-rewriter-training"


def audit_publication(root: Path, private_needles=()) -> dict[str, bytes]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contents = {}
    for name, expected in manifest["files"].items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("unsafe publication path")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"publication hash differs: {name}")
        contents[name] = data
    contents["manifest.json"] = (root / "manifest.json").read_bytes()
    for data in contents.values():
        if re.search(rb"sk-or-v1-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}", data) or any(
            needle and needle in data for needle in private_needles
        ):
            raise ValueError("private content in publication")
    required = {
        "supporting.json",
        "quality.json",
        "README.md",
        "DATA_LICENSE.md",
        *(
            f"{kind}/{split}.jsonl"
            for kind in ["data", "sft"]
            for split in ["train", "dev", "validation"]
        ),
    }
    if not required.issubset(contents):
        raise ValueError("publication is missing required data or source notices")
    support = json.loads(contents["supporting.json"])
    quality = json.loads(contents["quality.json"])
    review_names = {m["opus_review"]["report"] for m in support}
    if not review_names.issubset(contents):
        raise ValueError("missing exact review report")
    reviews = load_reviews([root / name for name in sorted(review_names)])
    ids, groups, total = set(), {}, 0
    for split in ["train", "dev", "validation"]:
        rows = [json.loads(x) for x in contents[f"data/{split}.jsonl"].splitlines()]
        guides = [json.loads(x) for x in contents[f"sft/{split}.jsonl"].splitlines()]
        meta = [m for m in support if m["split"] == split]
        if not (len(rows) == len(guides) == len(meta) == quality["splits"][split]):
            raise ValueError("publication split counts differ")
        total += len(rows)
        for row, guide, item in zip(rows, guides, meta, strict=True):
            if set(row) != {"input", "output", "mode", "checks"}:
                raise ValueError("publication row schema differs")
            case = Case(id=item["id"], **row, context=item["context"], language=item["language"])
            expected_guide = {
                "system": COMMON_SYSTEM + MODES[case.mode],
                "user": case.user_message(),
                "assistant": case.output,
            }
            if guide != expected_guide or digest(row) != item["row_sha256"]:
                raise ValueError("publication supervised text differs from verified row")
            if item["id"] in ids:
                raise ValueError("duplicate publication identity")
            ids.add(item["id"])
            previous = groups.setdefault(item["source_group"], split)
            if previous != split:
                raise ValueError("publication source group crosses splits")
            review = reviews.get((item["id"], digest(row)))
            saved = {k: v for k, v in item["opus_review"].items() if k != "report"}
            actual = {k: v for k, v in (review or {}).items() if k != "report"}
            if not review or review["verdict"] != "pass" or saved != actual:
                raise ValueError("publication row lacks its exact passing review")
            if item["mechanical_issues"] or item["human_reviewed"] is not False:
                raise ValueError("publication has unresolved checks or false human labels")
            if set(item["decisions"]) != {"labeling_engine." + k for k in DIMENSIONS}:
                raise ValueError("publication is missing native decision dimensions")
    if total != len(support) or Counter(x["split"] for x in support) != quality["splits"]:
        raise ValueError("publication contains foreign split metadata")
    return contents


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    personal = project / "data/rewriter/personal/calibration-10"
    needles = [x.encode() for x in [os.environ.get("OPENROUTER_API_KEY"), get_token()] if x]
    if personal.exists():
        needles.extend(
            json.loads(x)["input"].encode()
            for x in (personal / "cases.jsonl").read_text().splitlines()
        )
        needles.extend(
            x["source_id"].encode()
            for x in json.loads((personal / "supporting/cases.json").read_text())
        )
    verified = audit_publication(args.release, needles)
    contents = {f"{PREFIX}/{name}": data for name, data in verified.items()}
    contents["README.md"] = verified["README.md"]
    if args.prepare_only:
        print(json.dumps({"verified_files": len(contents), "prefix": PREFIX, "repo": args.repo}))
        return
    api = HfApi()
    info = api.repo_info(args.repo, repo_type="dataset")
    if info.private is not True:
        raise ValueError("Reviewed data must remain private")
    existing = api.list_repo_files(args.repo, repo_type="dataset", revision=info.sha)
    if any(name.startswith(PREFIX + "/") for name in existing):
        raise FileExistsError("Reviewed version already exists; never overwrite frozen data")
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        parent_commit=info.sha,
        commit_message="Add Opus-verified rewriter v2; retain rejected pilot as history",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
            for name, data in contents.items()
        ],
    )

    def verify(item):
        name, data = item
        downloaded = hf_hub_download(
            args.repo, name, repo_type="dataset", revision=commit.oid, force_download=True
        )
        if Path(downloaded).read_bytes() != data:
            raise ValueError(f"Uploaded bytes differ: {name}")
        return name, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(verify, contents.items()))
    receipt = {
        "repo_id": args.repo,
        "revision": commit.oid,
        "private": True,
        "prefix": PREFIX,
        "verified_file_count": len(hashes),
        "files": hashes,
        "url": f"https://huggingface.co/datasets/{args.repo}/tree/{commit.oid}/{PREFIX}",
    }
    output = project / "data/rewriter/releases/reviewed-v2.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
