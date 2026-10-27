"""Publish exactly the collected shared Liquid run and its mechanical report."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from finetuning_labs.rewriter.benchmark import load_bundle
from finetuning_labs.rewriter.evaluation import _load_generations

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/rewriter/evals/baseline-20261027"
REPO = "jayshah5696/finetuning-labs-rewriter"


def main():
    bundle = load_bundle()
    generations = _load_generations(bundle, RUN / "liquid")
    if len(generations) != len(bundle.cases):
        raise ValueError("Incomplete generation artifact set")
    names = [
        "modal.json",
        "liquid/run.json",
        "liquid/generations.json",
        "liquid-mechanical/mechanical.json",
        "liquid-mechanical/REPORT.md",
    ]
    names += [f"liquid/calls/{case.id}.json" for case in bundle.cases]
    contents = {
        f"evaluations/baseline-20261027/{name}": (RUN / name).read_bytes() for name in names
    }
    api = HfApi()
    info = api.dataset_info(REPO)
    if info.private is not True:
        raise ValueError("Diagnostic results must stay in the private dataset")
    existing = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.sha))
    if existing.intersection(contents):
        raise FileExistsError("The run is already published; review before replacing artifacts")
    commit = api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        parent_commit=info.sha,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
            for name, data in contents.items()
        ],
        commit_message="Save stock Liquid L4 outputs and mechanical-only findings",
    )

    def verify(item):
        name, data = item
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=commit.oid))
        if path.read_bytes() != data:
            raise ValueError(f"Published result differs: {name}")
        return name, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(verify, contents.items()))
    receipt = {
        "repo_id": REPO,
        "revision": commit.oid,
        "private": True,
        "verified_file_count": len(hashes),
        "files": hashes,
        "semantic_judge_status": "not run",
        "case_count": len(generations),
    }
    output = ROOT / "data/rewriter/releases/liquid-results.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
