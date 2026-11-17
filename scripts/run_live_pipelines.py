"""Historical expansion runner: execute live Arka pipelines and stage diverse-v2 and validation-v1 datasets."""

import json
from pathlib import Path

from finetuning_labs.rewriter.benchmark import load_bundle
from finetuning_labs.rewriter.diverse_expansion import (
    build_scenario_plan,
    load_openrouter_api_key,
    run_expansion_pipeline,
    stage_expansion_dataset,
)
from finetuning_labs.rewriter.expanded_schedule import get_expanded_coverage_schedule


def get_validation_plans() -> list[dict]:
    bundle_dir = Path("data/rewriter/expansion-benchmark-v1")
    bundle = load_bundle(bundle_dir)
    supporting_meta = json.loads((bundle_dir / "supporting/cases.json").read_text(encoding="utf-8"))
    meta_by_id = {m["id"]: m for m in supporting_meta}

    contrast_map = {
        "exp-contrast-01-t1-transcribe-low": "contrast-val-t1",
        "exp-contrast-02-t1-slack-medium": "contrast-val-t1",
        "exp-contrast-03-t1-polish-high": "contrast-val-t1",
        "exp-contrast-04-t1-prompt-custom": "contrast-val-t1",
        "exp-contrast-05-t2-transcribe-low": "contrast-val-t2",
        "exp-contrast-06-t2-slack-medium": "contrast-val-t2",
        "exp-contrast-07-t2-polish-high": "contrast-val-t2",
        "exp-contrast-08-t2-prompt-custom": "contrast-val-t2",
        "exp-contrast-09-t3-transcribe-low": "contrast-val-t3",
        "exp-contrast-10-t3-slack-medium": "contrast-val-t3",
        "exp-contrast-11-t3-polish-high": "contrast-val-t3",
        "exp-contrast-12-t3-prompt-custom": "contrast-val-t3",
        "exp-bench-041": "contrast-val-t4",
        "exp-bench-042": "contrast-val-t4",
        "exp-bench-043": "contrast-val-t4",
        "exp-bench-044": "contrast-val-t4",
        "exp-bench-049": "contrast-val-t5",
        "exp-bench-050": "contrast-val-t5",
        "exp-bench-051": "contrast-val-t5",
        "exp-bench-052": "contrast-val-t5",
    }

    plans = []
    for c in bundle.cases:
        meta = meta_by_id.get(c.id, {})
        rules = meta.get("rules", {})
        cid = c.id
        source_group = contrast_map.get(cid)

        family = meta.get("family")
        if not family:
            if source_group:
                family = "multilevel_contrasts"
            elif "code" in cid or cid in ["exp-bench-045", "exp-bench-053", "exp-bench-057"]:
                family = "developer_symbols"
            elif "file" in cid or cid in ["exp-bench-047", "exp-bench-054", "exp-bench-058"]:
                family = "workspace_context"
            elif "list" in cid or cid in ["exp-bench-048", "exp-bench-055", "exp-bench-059"]:
                family = "spoken_lists"
            else:
                family = "speech_repairs_and_isolation"

        tok_count = len(c.input.split())
        if tok_count > 600:
            length = "long"
        elif tok_count > 300:
            length = "medium"
        else:
            length = "short"

        p = build_scenario_plan(
            family=family,
            scenario_id=c.id,
            source_group=source_group,
            mode=c.mode,
            length=length,
            brief=c.input,
            target=c.output,
            context=c.context,
            language=c.language,
            split="val",
            required_any=rules.get("required_any"),
            forbidden=rules.get("forbidden"),
            literals=rules.get("literals"),
            identity=rules.get("identity", False),
            min_tokens_500=(length in ["medium", "long"] and len(c.input.split()) >= 300),
        )
        plans.append(p)
    return plans


def main():
    key = load_openrouter_api_key()
    assert key, "OPENROUTER_API_KEY could not be loaded!"
    print("OPENROUTER_API_KEY loaded successfully.")

    # 1. Run 136 Training Scenarios
    print("\n" + "=" * 70)
    print("STEP 1: Running Live Expansion Pipeline on 136 Training Scenarios")
    print("=" * 70)
    train_plans = get_expanded_coverage_schedule()
    train_run_dir = Path("data/rewriter/expansions/diverse-v2-run")
    train_stage_dir = Path("data/rewriter/expansions/diverse-v2")

    train_res = run_expansion_pipeline(
        train_run_dir,
        train_plans,
        workers=16,
        dry_run=False,
        resume=True,
    )
    print("Train Pipeline Result:")
    print(json.dumps(train_res, indent=2))

    train_staging = stage_expansion_dataset(
        train_run_dir,
        train_stage_dir,
        namespace="diverse-v2",
        dry_run=False,
    )
    print("Train Staging Result:")
    print(json.dumps(train_staging, indent=2))

    # 2. Run 60 Validation Benchmark Cases
    print("\n" + "=" * 70)
    print("STEP 2: Running Live Expansion Pipeline on 60 Validation Benchmark Cases")
    print("=" * 70)
    val_plans = get_validation_plans()
    val_run_dir = Path("data/rewriter/expansions/validation-v1-run")
    val_stage_dir = Path("data/rewriter/expansions/validation-v1")

    val_res = run_expansion_pipeline(
        val_run_dir,
        val_plans,
        workers=16,
        dry_run=False,
        resume=True,
    )
    print("Validation Pipeline Result:")
    print(json.dumps(val_res, indent=2))

    val_staging = stage_expansion_dataset(
        val_run_dir,
        val_stage_dir,
        namespace="validation-v1",
        dry_run=False,
    )
    print("Validation Staging Result:")
    print(json.dumps(val_staging, indent=2))

    print("\n" + "=" * 70)
    print("ALL LIVE EXPANSION PIPELINES COMPLETE & STAGED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
