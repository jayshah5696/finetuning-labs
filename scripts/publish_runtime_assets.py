"""Upload the exact Colab notebook, wheel, and config to the private dataset."""

import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "jayshah5696/finetuning-labs-rewriter"


def main():
    wheel = ROOT / "dist/finetuning_labs-0.1.0-py3-none-any.whl"
    sources = {
        "runtime/" + wheel.name: wheel,
        "runtime/rewriter.toml": ROOT / "configs/rewriter.toml",
        "notebooks/rewriter_baseline_l4.ipynb": ROOT / "notebooks/rewriter_baseline_l4.ipynb",
    }
    contents = {name: path.read_bytes() for name, path in sources.items()}
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    descriptor = {
        "code_commit": code_commit,
        "purpose": "Colab diagnostic baseline; no training or approved quality claims",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    contents["runtime/assets.json"] = (json.dumps(descriptor, indent=2) + "\n").encode()
    api = HfApi()
    info = api.dataset_info(REPO)
    if info.private is not True:
        raise ValueError("Runtime assets must stay in the private dataset")
    existing = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.sha))
    if existing.intersection(contents):
        raise FileExistsError("Runtime assets already exist; review before replacing them")
    commit = api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        parent_commit=info.sha,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
            for name, data in contents.items()
        ],
        commit_message="Add reproducible Colab L4 baseline assets",
    )
    for name, data in contents.items():
        downloaded = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=commit.oid))
        if downloaded.read_bytes() != data:
            raise ValueError(f"Runtime asset verification failed: {name}")
    receipt = {
        "repo_id": REPO,
        "revision": commit.oid,
        "private": True,
        "code_commit": code_commit,
        "verified_files": list(contents),
        "notebook_url": f"https://huggingface.co/datasets/{REPO}/blob/{commit.oid}/notebooks/rewriter_baseline_l4.ipynb",
    }
    output = ROOT / "data/rewriter/releases/runtime-assets.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
