"""Rebuild the five-model comparison from bound generation and judge calls."""

import csv
import json
from dataclasses import replace
from pathlib import Path

from finetuning_labs.rewriter.comparison import _bound_assessments, _record, _usage, paired_report
from finetuning_labs.rewriter.evaluation import (
    _load_generations,
    aggregate,
    bundle_identity,
    digest,
    evaluation_bundle,
    write_json,
)

ROOT = Path(__file__).resolve().parent
EVALS = ROOT.parent
SIZES = {s: EVALS / f"english-{s}-validation-20261101-a10g" for s in ["230m", "350m"]}
ARMS = {
    "230m_base": (SIZES["230m"] / "stock", ROOT / "230m-base-judge"),
    "230m_lora": (SIZES["230m"] / "lora", ROOT / "230m-lora-judge"),
    "350m_base": (SIZES["350m"] / "stock", ROOT / "350m-base-judge"),
    "350m_lora": (SIZES["350m"] / "lora", ROOT / "350m-lora-judge"),
    "luna6": (ROOT / "luna", ROOT / "luna-judge"),
}


def main():
    bundle = evaluation_bundle(ROOT / "bundle", english_only=True)
    setup = json.loads((ROOT / "setup.json").read_text())
    if len(bundle.cases) != 141 or bundle_identity(bundle) != setup["bundle_sha256"]:
        raise ValueError("Frozen 141-case evaluation changed")
    historical = evaluation_bundle(
        ROOT.parents[1] / "reviewed-v2/validation-bundle", english_only=True
    )
    if bundle.cases[:81] != historical.cases:
        raise ValueError("Historical 81 cases changed")
    subsets = {
        "all_141": bundle,
        "original_81": replace(bundle, cases=bundle.cases[:81]),
        "expansion_60": replace(bundle, cases=bundle.cases[81:]),
    }
    families = json.loads((ROOT / "expansion-families.json").read_text())
    for family in sorted({m["family"] for m in families}):
        ids = {m["id"] for m in families if m["family"] == family}
        subsets[family] = replace(bundle, cases=tuple(c for c in bundle.cases if c.id in ids))
    report = {
        "purpose": "English diagnostic development-validation comparison",
        "human_calibrated": False,
        "locked_test": False,
        "setup": setup,
        "pass_definition": "Deterministic checks and all five Sol criteria pass; denominator includes every scheduled case.",
        "critical_definition": "Number of cases with at least one Sol-identified factual omission, invention or meaning change. Missing reviews do not establish absence of errors.",
        "judge_pass_definition": "Sol verdict alone. A Sol pass does not waive explicit deterministic requirements such as identity, file tags or list formatting.",
        "length_definition": "Long case IDs from the frozen bundle manifest; the other bucket also includes medium cases.",
        "arms": {},
        "paired": {},
    }
    all_assessments, all_generations, judge_configs, prompts = {}, {}, [], []
    for name, (generation, judge) in ARMS.items():
        assessments, judge_calls, judge_config = _bound_assessments(bundle, generation, judge)
        generations = _load_generations(bundle, generation)
        spec = json.loads((generation / "run.json").read_text())["spec"]
        if name != "luna6":
            config = spec["config"]
            if (
                config["dataset_revision"] != setup["dataset_revision"]
                or config["source_bundle_sha256"] != setup["bundle_sha256"]
                or config["validation_count"] != 141
            ):
                raise ValueError("Liquid run used another dataset or validation selection")
            if name.endswith("_lora"):
                size = name.split("_")[0]
                if config["adapter_revision"] != setup["adapter_revisions"][size]:
                    raise ValueError("Liquid run used another adapter revision")
            for row in generations:
                if row["status"] != "completed":
                    continue
                if row["model"] != config["model"] or row["revision"] != config["revision"]:
                    raise ValueError("Actual Liquid model differs from its run configuration")
                expected_adapter = (
                    {"model": config["adapter_model"], "revision": config["adapter_revision"]}
                    if name.endswith("_lora")
                    else None
                )
                if row["adapter"] != expected_adapter:
                    raise ValueError("Actual Liquid adapter differs from the requested arm")
        prompts.append(spec["prompt_sha256"])
        judge_configs.append(judge_config)
        all_assessments[name] = {r["case_id"]: r for r in assessments}
        all_generations[name] = {r["case_id"]: r for r in generations}
        report["arms"][name] = {
            "config": spec["config"],
            "metrics": {
                group: aggregate(part, [all_assessments[name][c.id] for c in part.cases])
                for group, part in subsets.items()
            },
            "judge_and_rule_counts": {
                group: {
                    "sol_pass": sum(
                        all_assessments[name][c.id].get("judgment", {}).get("verdict") == "pass"
                        for c in part.cases
                    ),
                    "sol_pass_with_rule_failure": sum(
                        all_assessments[name][c.id].get("judgment", {}).get("verdict") == "pass"
                        and bool(all_assessments[name][c.id]["mechanical_failures"])
                        for c in part.cases
                    ),
                }
                for group, part in subsets.items()
            },
            "critical_errors": [r for r in assessments if r["critical_error"]],
            "failed_case_ids": [r["case_id"] for r in assessments if r["status"] != "pass"],
            "generation_usage": _usage(
                [_record(generation / "calls" / f"{c.id}.json") for c in bundle.cases]
            ),
            "judge_usage": _usage(judge_calls),
        }
    if len(set(prompts)) != 1 or any(c != judge_configs[0] for c in judge_configs):
        raise ValueError("Models did not share identical behavioral prompts and judge settings")
    report["judge_config"] = judge_configs[0]
    report["judge_rubric_sha256"] = digest(judge_configs[0]["rubric"])
    for size in SIZES:
        pair = paired_report(
            bundle,
            ARMS[size + "_base"][0],
            ARMS[size + "_lora"][0],
            ARMS[size + "_base"][1],
            ARMS[size + "_lora"][1],
        )
        report["paired"][size] = {
            k: pair[k] for k in ["scheduled_pairs", "pairs", "pooled_gain_percentage_points"]
        }
    write_json(ROOT / "comparison.json", report)
    with (ROOT / "case-results.csv").open("w", newline="") as stream:
        fields = ["case_id", "cohort", "mode", "input", "reference"]
        for name in ARMS:
            fields += [name + "_status", name + "_critical", name + "_output", name + "_critique"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(bundle.cases):
            row = {
                "case_id": case.id,
                "cohort": "original_81" if i < 81 else "expansion_60",
                "mode": case.mode,
                "input": case.input,
                "reference": case.output,
            }
            for name in ARMS:
                assessed, generated = all_assessments[name][case.id], all_generations[name][case.id]
                row.update(
                    {
                        name + "_status": assessed["status"],
                        name + "_critical": assessed["critical_error"],
                        name + "_output": generated["text"],
                        name + "_critique": assessed.get("judgment", {}).get("critique", ""),
                    }
                )
            writer.writerow(row)
    from render_report import render

    (ROOT / "comparison.html").write_text(
        render(report, all_assessments, all_generations, bundle), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    cohort: {
                        k: metrics["pooled"][k]
                        for k in [
                            "scheduled",
                            "counts",
                            "verified_pass_rate_pct",
                            "critical_error_count",
                        ]
                    }
                    for cohort, metrics in arm["metrics"].items()
                    if cohort in {"all_141", "original_81", "expansion_60"}
                }
                for name, arm in report["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
