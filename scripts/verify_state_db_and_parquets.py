"""Verify 100% decision cache hits in state.db and exact parity across parquet stages."""

import json
import sqlite3
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.decisions.client import DecisionClient
from arka.decisions.factory import resolve_decision_config
from arka.labeling.rubric import RubricLoader
from arka.pipeline.decision_stages import record_state, rubric_questions
from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash

from finetuning_labs.rewriter.diverse_expansion import load_openrouter_api_key

load_openrouter_api_key()


def verify_run_and_staged(
    run_name: str,
    staged_name: str,
    expected_retained: int,
):
    print(f"\n{'=' * 75}\nVerifying {run_name} <-> {staged_name}\n{'=' * 75}")
    base = Path(__file__).resolve().parents[1]
    run_dir = base / f"data/rewriter/expansions/{run_name}"
    staged_dir = base / f"data/rewriter/expansions/{staged_name}"

    writer = OutputWriter()

    # 1. Byte-for-byte check of 02p and 06 decision panels
    p02_path = run_dir / "runs/author/stages/02p_decision_panel/data.parquet"
    p06_path = run_dir / "runs/author/stages/06_decision_panel/data.parquet"
    assert p02_path.exists(), f"Missing {p02_path}"
    assert p06_path.exists(), f"Missing {p06_path}"
    p02_bytes = p02_path.read_bytes()
    p06_bytes = p06_path.read_bytes()
    assert p02_bytes == p06_bytes, (
        f"02p and 06 parquets are NOT byte-for-byte identical in {run_name}!"
    )
    print(
        f"[OK] 02p_decision_panel and 06_decision_panel are byte-for-byte identical ({len(p02_bytes)} bytes)"
    )

    # 2. 01_source check
    s1_path = run_dir / "runs/author/stages/01_source/data.parquet"
    assert s1_path.exists(), f"Missing {s1_path}"
    s1_records = writer.read_parquet(s1_path)
    for r in s1_records:
        assert len(r.payload.instruction) < 15000, f"Seed instruction for {r.id} >= 15,000 chars"
        assert r.content_hash == content_hash(r.payload), f"Content hash mismatch on seed {r.id}"
    print(
        f"[OK] 01_source/data.parquet: {len(s1_records)} seed records all < 15,000 chars with matching hashes"
    )

    # 3. Read 02_generate and 06_decision_panel records
    s2_records = writer.read_parquet(run_dir / "runs/author/stages/02_generate/data.parquet")
    s6_records = writer.read_parquet(p06_path)
    assert len(s2_records) == len(s6_records), (
        f"Count mismatch: 02_generate={len(s2_records)}, 06={len(s6_records)}"
    )

    # 4. Check staged dataset
    staged_viewer_path = staged_dir / "viewer/train.jsonl"
    staged_viewer = [
        json.loads(line)
        for line in staged_viewer_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(staged_viewer) == expected_retained, (
        f"Expected {expected_retained} retained rows in {staged_name}, found {len(staged_viewer)}"
    )
    print(f"[OK] Staged viewer train.jsonl has exactly {len(staged_viewer)} retained rows")

    candidates = [
        json.loads(line)
        for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    staged_pairs = {(r["input"], r["output"]) for r in staged_viewer}
    staged_indices = [
        i for i, c in enumerate(candidates) if (c["input"], c["output"]) in staged_pairs
    ]
    assert len(staged_indices) == expected_retained, (
        f"Matched {len(staged_indices)} candidates to staged pairs, expected {expected_retained}"
    )

    # 5. Connect to state.db and setup DecisionClient for cache key checking
    cfg = ConfigLoader().load(run_dir / "pipeline.yaml")
    rubric = RubricLoader().load(run_dir / "rubric.yaml")
    questions = rubric_questions(rubric)
    assert len(questions) == 6, f"Expected 6 rubric questions, got {len(questions)}"

    dec_cfg = resolve_decision_config(cfg.decisions, None)
    client = DecisionClient(dec_cfg)

    con = sqlite3.connect(run_dir / "state.db")
    cur = con.cursor()
    db_keys = {r[0] for r in cur.execute("SELECT cache_key FROM decisions_cache").fetchall()}
    con.close()

    # 6. Verify 100% of retained rows
    checked_rows = 0
    total_dimension_checks = 0
    for idx in staged_indices:
        cand = candidates[idx]
        rec = s6_records[idx]
        grec = s2_records[idx]

        # Content hash check
        expected_hash = content_hash(rec.payload)
        assert rec.content_hash == expected_hash, (
            f"Row {idx} hash mismatch in 06_decision_panel: {rec.content_hash} != {expected_hash}"
        )
        assert grec.content_hash == expected_hash, (
            f"Row {idx} hash mismatch in 02_generate: {grec.content_hash} != {expected_hash}"
        )

        # Released text matching
        assert rec.payload.instruction.endswith(cand["input"]), (
            f"Instruction mismatch for row {idx}"
        )
        assert rec.payload.response == cand["output"], f"Response mismatch for row {idx}"

        # Exact match in 02p and 06
        assert len(rec.scores.decisions) == 6, f"Row {idx} does not have exactly 6 decisions in 06"

        # Check state.db cache hits for record_state(rec)
        s_rec = record_state(rec)
        for q in questions:
            k = client._key(s_rec, q)
            assert k in db_keys, (
                f"Cache miss in state.db for row {idx} question {q.name} (record_state)"
            )
            total_dimension_checks += 1

        # Check state.db cache hits for json.dumps({"input": ..., "output": ...})
        s_io = json.dumps({"input": cand["input"], "output": cand["output"]}, ensure_ascii=False)
        for q in questions:
            k_io = client._key(s_io, q)
            assert k_io in db_keys, (
                f"Cache miss in state.db for row {idx} question {q.name} (json.dumps input/output)"
            )

        checked_rows += 1

    print(f"[OK] 100% of retained rows ({checked_rows}/{expected_retained}) verified!")
    print(
        f"[OK] 100% dimension cache hits: {total_dimension_checks}/{total_dimension_checks} (6 dimensions per row)"
    )


def main():
    verify_run_and_staged("diverse-v2-run", "diverse-v2", expected_retained=115)
    verify_run_and_staged("validation-v1-run", "validation-v1", expected_retained=60)
    print("\n" + "=" * 75)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY (100% CACHE HITS & PARQUET PARITY)")
    print("=" * 75)


if __name__ == "__main__":
    main()
