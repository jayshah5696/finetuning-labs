"""Publish the two exact hosted-model assessments and their paired comparison."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from finetuning_labs.rewriter.benchmark import load_bundle
from finetuning_labs.rewriter.comparison import paired_report

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/rewriter/evals/baseline-20261027"
REPO = "jayshah5696/finetuning-labs-rewriter"


def main():
    bundle = load_bundle()
    report = paired_report(
        bundle, RUN / "liquid", RUN / "luna", RUN / "liquid-judge", RUN / "luna-judge"
    )
    if json.loads((RUN / "comparison/comparison.json").read_text()) != report:
        raise ValueError("Comparison does not match the exact saved provider calls")
    names = [
        "luna/run.json",
        "luna/generations.json",
        "comparison/comparison.json",
        "comparison/REPORT.md",
        "comparison/REVIEW.md",
        "setup/luna.json",
        "setup/sol-liquid-first.json",
    ]
    names += [f"luna/calls/{case.id}.json" for case in bundle.cases]
    for model in ["liquid", "luna"]:
        names += [
            f"{model}-judge/{file}" for file in ["run.json", "assessments.json", "metrics.json"]
        ]
        names += [f"{model}-judge/calls/{case.id}.json" for case in bundle.cases]
    contents = {
        f"evaluations/baseline-20261027/{name}": (RUN / name).read_bytes() for name in names
    }
    api = HfApi()
    info = api.dataset_info(REPO)
    if info.private is not True:
        raise ValueError("The diagnostic dataset must remain private")
    existing = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.sha))
    if existing.intersection(contents):
        raise FileExistsError("Hosted results already exist; review before replacing them")
    commit = api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        parent_commit=info.sha,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
            for name, data in contents.items()
        ],
        commit_message="Add Luna baseline and matched Sol judgments for both models",
    )

    def verify(item):
        name, data = item
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=commit.oid))
        if path.read_bytes() != data:
            raise ValueError(f"Published bytes differ: {name}")
        return name, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(verify, contents.items()))
    receipt = {
        "repo_id": REPO,
        "revision": commit.oid,
        "private": True,
        "verified_file_count": len(hashes),
        "files": hashes,
        "judge_model": report["judge_config"]["model"],
        "human_calibrated": False,
        "comparison_url": f"https://huggingface.co/datasets/{REPO}/blob/{commit.oid}/evaluations/baseline-20261027/comparison/REPORT.md",
    }
    (ROOT / "data/rewriter/releases/hosted-results.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
