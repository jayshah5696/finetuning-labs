"""Run live openai/gpt-6.1-sol for sched-ws-med-02 and sched-ws-med-06 and update all reviews."""

import json
import os
import shutil
from pathlib import Path

from finetuning_labs.rewriter.diverse_expansion import load_openrouter_api_key
from finetuning_labs.rewriter.evaluation import write_json
from finetuning_labs.rewriter.scale_data import review_candidates

load_openrouter_api_key()
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is required!"

run_dir = Path("data/rewriter/expansions/diverse-v2-run")

# 1. Unlink existing call caches for sched-ws-med-02 (0078) and sched-ws-med-06 (0082)
for num in [78, 82]:
    cand_path = run_dir / f"sol-calls/{num:04d}.json"
    if cand_path.exists():
        print(f"Removing old call cache {cand_path}")
        cand_path.unlink()
    for retry in (run_dir / "sol-calls").glob(f"{num:04d}-retry-*.json"):
        print(f"Removing old retry cache {retry}")
        retry.unlink()

# 2. Run Round 1 Sol review
print("\nRunning Sol Round 1 review...")
summary = review_candidates(
    run_dir,
    prefix="sol",
    model="openai/gpt-6.1-sol",
    batch_size=1,
    max_tokens=4096,
    workers=8,
    retry_failed=True,
)
print("Round 1 review summary:", summary)

# Check Round 1 reviews
sol_file = run_dir / "sol-reviews.json"
sol_report = json.loads(sol_file.read_text(encoding="utf-8"))
revs_by_id = {r["id"]: r for r in sol_report.get("reviews", [])}

for tid in ["sched-ws-med-02", "sched-ws-med-06"]:
    r = revs_by_id.get(tid)
    print(f"\nRound 1 review for {tid}:")
    print(json.dumps(r, indent=2))

# Ensure sol_reviews.json also exists (synced copy)
shutil.copyfile(sol_file, run_dir / "sol_reviews.json")

# 3. Check and run Round 2 in adjudication
adj_dir = run_dir / "adjudication"
adj_calls = adj_dir / "sol-calls"
# Remove old sched-ws-med-06 in adjudication/sol-calls (0019.json)
p_adj_06 = adj_calls / "0019.json"
if p_adj_06.exists():
    print(f"\nRemoving old adjudication call {p_adj_06}")
    p_adj_06.unlink()
for retry in adj_calls.glob("0019-retry-*.json"):
    retry.unlink()

# Update adjudication generation candidates and supporting with current metadata
adj_gen = adj_dir / "generation"
adj_cands = [json.loads(l) for l in (adj_gen / "candidates.jsonl").read_text().splitlines()]
adj_supp = json.loads((adj_gen / "supporting.json").read_text())

# Find sched-ws-med-06 in adj
main_cands = [json.loads(l) for l in (run_dir / "candidates.jsonl").read_text().splitlines()]
main_supp = json.loads((run_dir / "supporting.json").read_text())
main_by_id = {m["id"]: (c, m) for c, m in zip(main_cands, main_supp)}

for i, m in enumerate(adj_supp):
    if m["id"] in main_by_id:
        c, sm = main_by_id[m["id"]]
        adj_cands[i] = c
        adj_supp[i] = sm

(adj_gen / "candidates.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in adj_cands)
)
write_json(adj_gen / "supporting.json", adj_supp)
write_json(adj_gen / "scored-supporting.json", adj_supp)

print("\nRunning Sol Round 2 adjudication review...")
summary_r2 = review_candidates(
    adj_dir,
    prefix="sol",
    model="openai/gpt-6.1-sol",
    batch_size=1,
    max_tokens=4096,
    workers=8,
    retry_failed=True,
    system_prompt=(
        "You are an independent second-round expert reviewer evaluating rewritten conversational requests.\n"
        "Provide an independent, fresh critical analysis evaluating the candidate rewrite against the criteria.\n"
        "Strictly assess input fidelity, output quality, pair validity, and mode requirements.\n"
        "Give a concrete critique for EVERY pair, citing specific details."
    ),
)
print("Round 2 review summary:", summary_r2)

adj_sol_file = adj_dir / "sol-reviews.json"
shutil.copyfile(adj_sol_file, adj_dir / "sol_reviews.json")
shutil.copyfile(adj_sol_file, run_dir / "sol-reviews-round2.json")

adj_report = json.loads(adj_sol_file.read_text(encoding="utf-8"))
adj_revs = {r["id"]: r for r in adj_report.get("reviews", [])}
if "sched-ws-med-06" in adj_revs:
    print("\nRound 2 review for sched-ws-med-06:")
    print(json.dumps(adj_revs["sched-ws-med-06"], indent=2))
