"""Current verification script: check public downloads, exact rows and Dataset Viewer splits without credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from finetuning_labs.rewriter.hf_data import PAIR_COLUMNS, PROJECT, SFT_COLUMNS, _sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/public-hub-check.json")
    args = parser.parse_args()
    receipts = json.loads((PROJECT / "data/rewriter/releases/public-parquet.json").read_text())
    receipts = {r["repo_id"]: r for r in receipts["repositories"]}
    sources, _, _ = _sources()
    api, results = HfApi(token=False), []
    with httpx.Client(base_url="https://datasets-server.huggingface.co", timeout=120) as client:
        for repo, rows, guides, *_ in sources:
            receipt = receipts[repo]
            info = api.repo_info(repo, repo_type="dataset")
            assert not info.private and info.sha == receipt["revision"], "Public revision changed"
            files = api.list_repo_files(repo, repo_type="dataset", revision=info.sha)
            assert set(files) == set(receipt["files"]) | {".gitattributes"}, "Unexpected Hub files"
            for name, expected in receipt["files"].items():
                path = hf_hub_download(
                    repo, name, repo_type="dataset", revision=info.sha, token=False
                )
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, name
            split_response = client.get("/splits", params={"dataset": repo})
            split_response.raise_for_status()
            split_data = split_response.json()
            assert not split_data.get("pending") and not split_data.get("failed"), split_data
            expected_splits = {(c, s) for c in ("default", "sft") for s in rows}
            assert {(s["config"], s["split"]) for s in split_data["splits"]} == expected_splits
            checks = []
            for config, expected_rows, columns in [
                ("default", rows, PAIR_COLUMNS),
                ("sft", guides, SFT_COLUMNS),
            ]:
                loaded = load_dataset(repo, config, revision=info.sha, token=False)
                assert set(loaded) == set(expected_rows)
                for split, expected in expected_rows.items():
                    actual = loaded[split]
                    assert actual.column_names == list(columns)
                    assert actual.to_list() == expected, f"Changed rows: {repo}/{config}/{split}"
                    response = client.get(
                        "/rows",
                        params={
                            "dataset": repo,
                            "config": config,
                            "split": split,
                            "offset": 0,
                            "length": 1,
                        },
                    )
                    response.raise_for_status()
                    viewer = response.json()
                    assert viewer["num_rows_total"] == len(expected)
                    assert [f["name"] for f in viewer["features"]] == list(columns)
                    assert viewer["rows"][0]["row"] == expected[0], "Viewer differs from source"
                    checks.append(
                        {
                            "config": config,
                            "split": split,
                            "rows": len(actual),
                            "viewer_status": 200,
                        }
                    )
            results.append(
                {
                    "repo_id": repo,
                    "revision": info.sha,
                    "public": True,
                    "file_count": len(files),
                    "checks": checks,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "authentication": "none",
        "all_downloaded_rows_equal_git_sources": True,
        "all_current_file_hashes_match_receipt": True,
        "repositories": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
