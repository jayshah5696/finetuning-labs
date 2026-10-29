"""Replace reviewed-but-defective new pairs, then rebuild from the unchanged Git baseline.

Refuse a changed selection, incompatible replacement, overlap or incomplete verification.
Keep original verdicts and both selections. Audit the rebuilt release before the atomic swap.
This recovery uses the saved Arka run, including its native Parquet artifacts; it makes no API calls.
"""

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

from finetuning_labs.rewriter import boost_data
from finetuning_labs.rewriter.english_critical import QUOTAS, verify_direction_balance
from finetuning_labs.rewriter.evaluation import digest, write_json

REPLACEMENTS = {
    "arka-english-main-0-prompt-medium-0162": "arka-english-main-0-prompt-medium-0152",
    "arka-english-main-1-prompt-long-0037": "arka-english-main-3-prompt-long-0007",
    "arka-english-main-0-transcribe-short-0255": "arka-english-quality-5-transcribe-short-0185",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    args = parser.parse_args()
    run, current = args.run.resolve(), boost_data.CURRENT
    spec = json.loads((run / "spec.json").read_text())
    original_contents = {}
    for name in spec["base_files"]:
        content = (current / name).read_bytes()
        if name.endswith("/train.jsonl"):
            content = b"".join(content.splitlines(keepends=True)[: spec["base_counts"]["train"]])
        original_contents[name] = content
    boost_data.verify_holdouts(original_contents, spec["base_files"])
    entries = json.loads((run / "retained.json").read_text())
    actual = [json.loads(line) for line in (current / "data/train.jsonl").read_text().splitlines()]
    assert actual[-500:] == [e["row"] for e in entries], (
        "Current append differs from saved selection"
    )
    candidates = {e["meta"]["id"]: e for e in json.loads(args.candidates.read_text())}
    blocked = boost_data.protected_index()
    revised, changes = [], []
    for entry in entries:
        old = entry["meta"]
        if old["id"] not in REPLACEMENTS:
            revised.append(entry)
            continue
        replacement = candidates[REPLACEMENTS[old["id"]]]
        new = replacement["meta"]
        assert all(old[k] == new[k] for k in ["mode", "length", "focus", "identity", "language"])
        assert not blocked.matches(replacement["row"]["input"])
        assert not blocked.matches(replacement["row"]["output"])
        revised.append(replacement)
        changes.append(
            {
                "excluded_id": old["id"],
                "excluded_row_sha256": old["row_sha256"],
                "replacement_id": new["id"],
                "replacement_row_sha256": new["row_sha256"],
                "reason": (
                    "Explicit spoken correction is left as a repair cue"
                    if old["mode"] == "/transcribe"
                    else "Explanation request contains an unqualified prohibition on answering"
                ),
                "reviewer_model": "gpt-6.1-sol",
                "previous_verdicts_preserved": True,
            }
        )
    assert len(changes) == len(REPLACEMENTS)
    boost_data.validate_selection(revised, quotas=QUOTAS)
    assert sum(e["meta"]["identity"] for e in revised) == 125
    preserved = ["retained.json", "direction-balance.json", "final-overlap-check.json"]
    original = {name: (run / name).read_bytes() for name in preserved}
    backup = run / "before-quality-backfill-retained.json"
    assert not backup.exists(), "This recovery has already started; inspect its saved artifacts"
    backup.write_bytes(original["retained.json"])
    write_json(run / "quality-backfills.json", {"replacements": changes})
    write_json(run / "retained.json", revised)
    sha = digest([e["row"] for e in revised])
    balance = json.loads(original["direction-balance.json"])
    balance.update(selected_rows_sha256=sha, quality_backfills=changes)
    verify_direction_balance(revised, balance)
    write_json(run / "direction-balance.json", balance)
    write_json(
        run / "final-overlap-check.json", {"rows": 500, "new_to_new_overlap": 0, "rows_sha256": sha}
    )
    swapped = False
    try:
        root = Path(__file__).resolve().parents[3]
        archive = subprocess.check_output(
            [
                "git",
                "archive",
                args.baseline_commit,
                "projects/finetuning-labs/data/rewriter/current",
            ],
            cwd=root,
        )
        base = run / "quality-rebuild-base"
        base.mkdir(exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(base, filter="data")
        rebuilt = base / "projects/finetuning-labs/data/rewriter/current"
        boost_data.CURRENT = rebuilt
        boost_data.apply_boost(run)
        assert [
            json.loads(line) for line in (rebuilt / "data/train.jsonl").read_text().splitlines()
        ][-500:] == [e["row"] for e in revised]
        boost_data.replace_dataset(rebuilt, current)
        swapped = True
    finally:
        boost_data.CURRENT = current
        if not swapped:
            for name, content in original.items():
                (run / name).write_bytes(content)
    print(
        json.dumps({"replaced": len(changes), "selected_rows_sha256": sha, "current": str(current)})
    )


if __name__ == "__main__":
    main()
