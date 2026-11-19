"""Historical migration script: sync Arka DecisionClient, state.db decisions_cache, and parquet stages for diverse-v2-run and validation-v1-run."""

import json
import os
import runpy
import shutil
from datetime import UTC, datetime
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.decisions.client import DecisionClient
from arka.decisions.factory import resolve_decision_config
from arka.labeling.rubric import RubricLoader
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.decision_stages import record_state, rubric_questions
from arka.pipeline.output import OutputWriter
from arka.records.identity import config_hash, content_hash, file_hash, record_id
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)

from finetuning_labs.rewriter.arka_pipeline import header
from finetuning_labs.rewriter.benchmark import MODES, Case
from finetuning_labs.rewriter.diverse_expansion import (
    build_rules,
    load_openrouter_api_key,
    validate_diverse_candidate,
)
from finetuning_labs.rewriter.evaluation import digest, write_json
from finetuning_labs.rewriter.expanded_schedule import get_expanded_coverage_schedule
from finetuning_labs.rewriter.training import Example, _tokenizer, token_audit

get_validation_plans = runpy.run_path(
    str(Path(__file__).resolve().parent / "run_live_pipelines.py")
)["get_validation_plans"]


def sync_run_dir(run_dir: Path, plans: list[dict], namespace: str):
    print(f"\n{'=' * 70}\nSyncing {namespace} in {run_dir}\n{'=' * 70}")
    cfg = ConfigLoader().load(run_dir / "pipeline.yaml")
    cfg_h = config_hash(cfg)
    rubric = RubricLoader().load(run_dir / "rubric.yaml")
    questions = rubric_questions(rubric)
    dec_cfg = resolve_decision_config(cfg.decisions, None)
    checkpoint_mgr = CheckpointManager(run_dir / "state.db")
    client = DecisionClient(dec_cfg, checkpoint=checkpoint_mgr)

    # 1. Load candidates and supporting metadata
    cand_file = run_dir / "candidates.jsonl"
    candidates = [
        json.loads(line)
        for line in cand_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    supp_file = run_dir / "supporting.json"
    metadata = json.loads(supp_file.read_text(encoding="utf-8"))

    plan_by_id = {p["id"]: p for p in plans}
    tokenizer = _tokenizer()

    # Update candidate rows to match plan targets / briefs where applicable
    for idx, meta in enumerate(metadata):
        pid = meta["id"]
        plan = plan_by_id[pid]
        # Sync plan data into meta
        for k in ["brief", "target", "context", "required_any", "forbidden", "literals"]:
            if k in plan:
                meta[k] = plan[k]

        # If candidate row needs updating to match plan
        if pid in ["sched-ws-med-02", "sched-ws-med-06"]:
            candidates[idx]["input"] = plan["brief"]
            candidates[idx]["output"] = plan["target"]
            candidates[idx]["mode"] = plan["mode"]

        row = candidates[idx]
        issues = validate_diverse_candidate(row["input"], row["output"], plan.get("context"), plan)
        meta["mechanical_issues"] = issues

        case_rules = build_rules(plan)
        case = Case(
            id=pid,
            input=row["input"],
            output=row["output"],
            mode=row["mode"],
            checks=MODES[row["mode"]]
            + " Preserve every fact, request, condition and quoted literal.",
            context=plan.get("context", {}),
            language=plan.get("language", "en"),
            rules=case_rules,
        )
        row["checks"] = case.checks
        meta["row_sha256"] = digest(row)

        audit_res = token_audit(
            Example(
                case,
                "train" if plan.get("split") == "train" else "validation",
                plan.get("source_group", pid),
                plan.get("length", "short"),
                "independent",
                pid,
            ),
            tokenizer,
        )
        meta["tokens"] = {k: v for k, v in audit_res.items() if k not in {"input_ids", "labels"}}

    # 2. Evaluate all rows through DecisionClient and sync state.db
    print("Evaluating decisions across all rows...")
    eval_decisions_by_id = {}
    for idx, (cand, meta) in enumerate(zip(candidates, metadata, strict=True)):
        pid = meta["id"]
        plan = plan_by_id[pid]
        inst = header(plan) + cand["input"]
        resp = cand["output"]

        payload = ConversationPayload(instruction=inst, response=resp)
        rec = ConversationRecord(
            id="dummy",
            content_hash="dummy",
            source=RecordSource(type="generated"),
            lineage=RecordLineage(root_id="dummy", parent_ids=[]),
            payload=payload,
            scores=RecordScores(),
            config_hash=cfg_h,
            created_at=datetime.now(UTC).isoformat(),
        )
        s = record_state(rec)
        result = client.decide(s, questions)
        eval_decisions_by_id[pid] = result.answers

        # Also populate cache with json.dumps({"input": ..., "output": ...})
        s_io = json.dumps({"input": cand["input"], "output": cand["output"]}, ensure_ascii=False)
        for q in questions:
            ans_serialized = result.answers[q.name].model_dump_json()
            k_io = client._key(s_io, q)
            checkpoint_mgr.save_decision(k_io, ans_serialized)

        meta["decisions"] = {
            q.name: result.answers[q.name].model_dump(mode="json") for q in questions
        }
        if idx % 20 == 0 or idx == len(candidates) - 1:
            print(
                f"  Processed {idx + 1}/{len(candidates)} ({pid}, cache hits: {result.cache_hits})"
            )

    # 3. Build updated seeds.jsonl
    print("Writing seeds.jsonl...")
    seeds = []
    for plan in plans:
        seed_dict = {
            "id": plan["id"],
            "mode": plan["mode"],
            "language": plan["language"],
            "context": plan["context"],
            "length": plan["length"],
            "header": header(plan),
            "mode_contract": MODES[plan["mode"]],
            "brief": plan["brief"],
        }
        if plan.get("target"):
            test_seed = {**seed_dict, "reference_target": plan["target"]}
            if len(json.dumps(test_seed, ensure_ascii=False)) < 15000:
                seed_dict["reference_target"] = plan["target"]
        seeds.append(
            {
                "instruction": json.dumps(seed_dict, ensure_ascii=False),
                "response": "",
            }
        )
    seeds_file = run_dir / "seeds.jsonl"
    seeds_file.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in seeds), encoding="utf-8"
    )
    seed_file_h = file_hash(seeds_file)

    # 4. Build 01_source/data.parquet
    print("Writing 01_source/data.parquet...")
    now_iso = datetime.now(UTC).isoformat()
    source_records = []
    for seed in seeds:
        payload = ConversationPayload(instruction=seed["instruction"], response="")
        rid = record_id(payload)
        chash = content_hash(payload)
        rec = ConversationRecord(
            id=rid,
            content_hash=chash,
            source=RecordSource(type="seed", seed_file_hash=seed_file_h),
            lineage=RecordLineage(root_id=rid, parent_ids=[]),
            payload=payload,
            scores=RecordScores(),
            config_hash=cfg_h,
            created_at=now_iso,
        )
        source_records.append(rec)

    stages_dir = run_dir / "runs/author/stages"
    writer = OutputWriter()
    writer.write_parquet(source_records, stages_dir / "01_source/data.parquet")

    # 5. Build 02_generate/data.parquet, 02p_decision_panel, 06_decision_panel
    print("Writing 02_generate and decision panel parquets...")
    gen_records = []
    panel_records = []
    for srec, plan, cand in zip(source_records, plans, candidates, strict=True):
        inst = header(plan) + cand["input"]
        resp = cand["output"]
        payload = ConversationPayload(instruction=inst, response=resp)
        lineage = RecordLineage(
            root_id=srec.id, parent_ids=[srec.id], operator="prompt_based", round=1, depth=1
        )
        source = RecordSource(
            type="generated", source_hash=srec.content_hash, seed_file_hash=seed_file_h
        )
        gid = record_id(payload, lineage)
        gchash = content_hash(payload)

        # Update meta record_id
        for m in metadata:
            if m["id"] == plan["id"]:
                m["record_id"] = gid
                break

        grec = ConversationRecord(
            id=gid,
            content_hash=gchash,
            source=source,
            lineage=lineage,
            payload=payload,
            scores=RecordScores(),
            config_hash=cfg_h,
            created_at=now_iso,
        )
        gen_records.append(grec)

        answers = eval_decisions_by_id[plan["id"]]
        prec = grec.model_copy(deep=True)
        prec.scores.decisions = {q.name: answers[q.name] for q in questions}
        panel_records.append(prec)

    writer.write_parquet(gen_records, stages_dir / "02_generate/data.parquet")
    p02_path = stages_dir / "02p_decision_panel/data.parquet"
    writer.write_parquet(panel_records, p02_path)

    p06_path = stages_dir / "06_decision_panel/data.parquet"
    p06_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p02_path, p06_path)

    # 6. Save updated candidates and supporting files
    print("Saving updated candidates.jsonl and supporting.json...")
    cand_str = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in candidates)
    (run_dir / "candidates.jsonl").write_text(cand_str, encoding="utf-8")
    write_json(run_dir / "supporting.json", metadata)
    write_json(run_dir / "candidate-supporting.json", metadata)

    gen_dir = run_dir / "generation"
    if gen_dir.exists():
        (gen_dir / "candidates.jsonl").write_text(cand_str, encoding="utf-8")
        write_json(gen_dir / "supporting.json", metadata)
        if (gen_dir / "scored-supporting.json").exists():
            write_json(gen_dir / "scored-supporting.json", metadata)

    print(f"Sync complete for {namespace}!")


def main():
    load_openrouter_api_key()
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY required!"

    train_plans = get_expanded_coverage_schedule()
    sync_run_dir(
        Path("data/rewriter/expansions/diverse-v2-run"),
        train_plans,
        "diverse-v2",
    )

    val_plans = get_validation_plans()
    sync_run_dir(
        Path("data/rewriter/expansions/validation-v1-run"),
        val_plans,
        "validation-v1",
    )


if __name__ == "__main__":
    main()
