"""Package only the shared calibration bundle; never upload the working tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .benchmark import audit, load_bundle, render_review

CARD = """---
language:
- en
- de
- pt
license: other
license_name: mixed-source-draft
license_link: DATA_LICENSE.md
task_categories:
- text-generation
tags:
- text-rewriting
- evaluation
- draft
configs:
- config_name: default
  data_files:
  - split: calibration
    path: data/calibration.jsonl
---
# Finetuning Labs rewriter calibration draft

68 input/output examples across `/slack`, `/transcribe`, `/prompt`, and `/polish`.
The data has four columns: `input`, `output`, `mode`, and `checks`.
Eight long inputs have 1,094 to 1,865 pinned Liquid tokenizer tokens. They represent
four fictional source documents, each used in two modes. Keep related source groups together.

This is a calibration draft. Human review is pending. Reference answers and the 156 candidate
fixture labels are authored expectations. This is not an independent locked test or a training
split. No personal TypeWhisper history, audio, credentials, or personal reference drafts are
included. Diagnostic model outputs, when present under `evaluations/`, use an uncalibrated judge.

`bundle/` contains the exact shared source bundle, context, literal rules, judge rubric, source
snapshot, and license. `review.html` shows complete examples. Extra metadata does not enter
the four-column dataset or supervised text. Consult `DATA_LICENSE.md` before redistribution.
"""

LICENSE_NOTE = """# Data provenance and license

Thirty authored fixtures are imported from TypeWhisper at revision
`c878eced59fc4ccc7755ccc5f30b1fbcc62c1939`. Their repository includes the GNU GPL version 3.
The complete original license is preserved at `bundle/supporting/TYPEWHISPER-LICENSE.txt`.
No separate fixture license grant was verified. The imported rows are not relicensed as Apache.

The other 38 examples and supporting candidate fixtures were authored for this project.
No additional public redistribution license is asserted for those new records in this private
draft. Source identity and grouping are preserved in `bundle/supporting/cases.json`.
Personal history entries are excluded. This draft is for review and diagnostic evaluation.
"""


def package_dataset(source: Path, destination: Path, *, code_commit: str) -> dict:
    bundle = load_bundle(source)
    report = audit(bundle)
    if report["mechanical_failures"]:
        raise ValueError("Mechanical fixture expectations failed")
    if any(c.source_kind == "personal_transcription" for c in bundle.cases):
        raise ValueError("Personal data cannot enter the shared Hub package")
    destination.mkdir(parents=True, exist_ok=False)
    allowed = {"manifest.json", "review/human-labels.csv", *bundle.manifest["files"]}
    for relative in sorted(allowed):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "personal" in path.parts:
            raise ValueError("Unsafe bundle path")
        original = source / path
        if original.is_symlink() or not original.is_file():
            raise ValueError("Bundle files must be regular files")
        target = destination / "bundle" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    (destination / "data").mkdir()
    shutil.copyfile(source / "cases.jsonl", destination / "data/calibration.jsonl")
    (destination / "README.md").write_text(CARD)
    (destination / "DATA_LICENSE.md").write_text(LICENSE_NOTE)
    (destination / "review.html").write_text(render_review(bundle))
    files = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "status": "draft",
        "human_review": "pending",
        "code_commit": code_commit,
        "case_count": len(bundle.cases),
        "private": True,
        "files": files,
    }
    (destination / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def upload_dataset(package: Path, repo_id: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    manifest = json.loads((package / "release.json").read_text())
    files = {
        **manifest["files"],
        "release.json": hashlib.sha256((package / "release.json").read_bytes()).hexdigest(),
    }
    actual = {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("Hub package has unlisted files; refusing upload")
    for name, expected in files.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Hub package file changed: {name}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    if api.dataset_info(repo_id).private is not True:
        raise ValueError("Expected a private draft repository; refusing public upload")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=package,
        allow_patterns=list(files),
        commit_message="Upload rewriter calibration draft",
    )
    for name, expected in files.items():
        downloaded = Path(hf_hub_download(repo_id, name, repo_type="dataset", revision=commit.oid))
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Uploaded bytes differ: {name}")
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "revision": commit.oid,
        "private": True,
        "verified_file_count": len(files),
        "case_count": manifest["case_count"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--bundle", type=Path, default=Path("data/rewriter/benchmark-draft"))
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--code-commit", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--package", type=Path, required=True)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "package":
        result = package_dataset(args.bundle, args.output, code_commit=args.code_commit)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    else:
        result = upload_dataset(args.package, args.repo_id)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
