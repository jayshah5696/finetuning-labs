"""Judge matched English runs and export stock, LoRA and hosted-model comparisons."""

import argparse
import csv
import json
import os
import tomllib
from collections import Counter
from pathlib import Path

from finetuning_labs.rewriter.benchmark import judge_packet, validate_judgment
from finetuning_labs.rewriter.comparison import _record, markdown, paired_report
from finetuning_labs.rewriter.evaluation import (
    _load_generations,
    bundle_identity,
    digest,
    evaluation_bundle,
    hosted_backend,
    judge_cases,
    write_json,
)

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--hosted-run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arm", choices=["stock", "lora"], action="append")
    args = parser.parse_args()
    if not args.run and not args.hosted_run:
        parser.error("Provide a collected --run or a saved --hosted-run")
    bundle_path = PROJECT / "data/rewriter/reviewed-v2/validation-bundle"
    bundle = evaluation_bundle(bundle_path, english_only=True)
    settings = tomllib.loads((PROJECT / "configs/rewriter.toml").read_text())
    config = settings["judge"] | {
        "rubric": (bundle_path / "supporting/judge-system.txt").read_text()
    }
    judge = hosted_backend(config, judge=True)
    arms, comparisons, case_reviews = [], [], {}
    runs = [(run, args.arm or ["stock", "lora"], True) for run in args.run]
    runs += [(run, ["luna"], False) for run in args.hosted_run]
    for run, selected_arms, modal_run in runs:
        if modal_run:
            receipt = json.loads((run / "modal.json").read_text())
            if receipt["status"] not in {"submitted", "collected"}:
                raise ValueError(f"The GPU run was not submitted successfully: {run}")
        for arm in selected_arms:
            # This rejects missing, incomplete, modified, or incorrectly bound arm records.
            generations = _load_generations(bundle, run / arm)
            destination = run / f"{arm}-judge"
            calls = destination / "calls"
            calls.mkdir(parents=True, exist_ok=True)
            # Match the whole packet and judge config, not just case ID or output text.
            sources = {}
            for path in sorted((PROJECT / "data/rewriter/evals").glob("*/*-judge/calls/*.json")):
                if path.parent == calls:
                    continue
                record = _record(path)
                if record["status"] != "completed" or record["request"]["config"] != config:
                    continue
                judgment = record["result"].get("judgment")
                if judgment is None:
                    continue
                validate_judgment(record["request"]["packet"]["case_id"], judgment)
                sources.setdefault(record["request_sha256"], (path, record))
            reused = []
            for case, generation in zip(bundle.cases, generations, strict=True):
                target = calls / f"{case.id}.json"
                if generation["status"] != "completed" or target.exists():
                    continue
                request = {"packet": judge_packet(case, generation["text"]), "config": config}
                cached = sources.get(digest(request))
                if cached is None:
                    continue
                source, record = cached
                if record["request"] != request:
                    raise ValueError("Cached packet differs despite its matching request hash")
                target.write_bytes(source.read_bytes())
                reused.append(
                    {
                        "case_id": case.id,
                        "source": str(source.relative_to(PROJECT)),
                        "request_sha256": record["request_sha256"],
                        "result_sha256": record["result_sha256"],
                    }
                )
            # Preserve the initial reuse receipt when resuming a partially judged arm.
            reuse_path = destination / "reuse.json"
            if reused or not reuse_path.exists():
                previous = json.loads(reuse_path.read_text()) if reuse_path.exists() else []
                write_json(reuse_path, previous + reused)
            print(json.dumps({"run": run.name, "arm": arm, "new_reused_calls": len(reused)}))
            rows = judge_cases(bundle, run / arm, destination, config, judge, workers=args.workers)
            metrics = json.loads((destination / "metrics.json").read_text())
            generation_config = json.loads((run / arm / "run.json").read_text())["spec"]["config"]
            label = generation_config["model"].split("/")[-1] + " " + arm
            if label in case_reviews:
                raise ValueError(f"Duplicate model label: {label}")
            case_reviews[label] = {
                case.id: (generation, assessment)
                for case, generation, assessment in zip(
                    bundle.cases, generations, rows, strict=True
                )
            }
            arms.append(
                {
                    "run": run.name,
                    "arm": arm,
                    "config": generation_config,
                    "metrics": metrics,
                    "critical_errors": [r for r in rows if r["critical_error"]],
                    "failed_case_ids": [r["case_id"] for r in rows if r["status"] != "pass"],
                }
            )
        if set(selected_arms) == {"stock", "lora"}:
            report = paired_report(
                bundle, run / "stock", run / "lora", run / "stock-judge", run / "lora-judge"
            )
            write_json(run / "comparison/comparison.json", report)
            (run / "comparison/REPORT.md").write_text(markdown(report))
            comparisons.append({"run": run.name, "pairs": report["pairs"]})
    result = {
        "bundle_sha256": bundle_identity(bundle),
        "validation_count": len(bundle.cases),
        "case_ids": [case.id for case in bundle.cases],
        "judge_config": config,
        "human_calibrated": False,
        "locked_test": False,
        "arms": arms,
        "comparisons": comparisons,
    }
    write_json(args.output / "comparison.json", result)
    lines = [
        "# English stock, LoRA and Luna comparison",
        "",
        "All arms use the same 81 English validation cases, prompts, 2,048-token output budget,",
        "and Sol rubric. Liquid models use greedy decoding. Hosted Luna uses reasoning effort none",
        "and provider-default sampling. Every scheduled case stays in the denominator.",
        "These are model judgments on development validation. Human verification and an unseen test are pending.",
        "",
        "| Model | Arm | Passes | Critical errors | Truncated | Identity unchanged | Long passes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        metrics = arm["metrics"]
        pooled = metrics["pooled"]
        long = metrics["by_length"]["long"]
        lines.append(
            f"| {arm['config']['model']} | {arm['arm']} | {pooled['counts']['pass']}/81 "
            f"({pooled['verified_pass_rate_pct']:.1f}%) | {pooled['critical_error_count']} | "
            f"{pooled['counts']['truncated']} | "
            f"{pooled['identity_unchanged']}/{pooled['identity_count']} | "
            f"{long['counts']['pass']}/{long['scheduled']} |"
        )
    lines += [
        "",
        "| Mode | "
        + " | ".join(a["config"]["model"].split("/")[-1] + " " + a["arm"] for a in arms)
        + " |",
        "| --- | " + " | ".join("---:" for _ in arms) + " |",
    ]
    for mode in ["/slack", "/transcribe", "/prompt", "/polish"]:
        values = [a["metrics"]["per_mode"][mode] for a in arms]
        lines.append(
            f"| {mode} | "
            + " | ".join(f"{v['counts']['pass']}/{v['scheduled']}" for v in values)
            + " |"
        )
    lines += [
        "",
        "Critical errors count outputs with changed, invented, or omitted meaning. Incomplete outputs",
        "are reported separately and have no semantic verdict. An unchanged identity is an exact match.",
        "The full configurations, adapter revisions, assessments and exact cached-judgment sources are saved alongside this report.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines))
    # One source row per case, with every model's complete output and assessment beside it.
    combined = []
    for case in bundle.cases:
        row = {
            "case_id": case.id,
            "mode": case.mode,
            "input": case.input,
            "reference": case.output,
            "checks": case.checks,
            "context": json.dumps(case.context, ensure_ascii=False, sort_keys=True),
        }
        for label, reviews in case_reviews.items():
            generation, assessment = reviews[case.id]
            judgment = assessment.get("judgment", {})
            row.update(
                {
                    label + " output": generation["text"],
                    label + " status": assessment["status"],
                    label + " critical_errors": json.dumps(
                        judgment.get("critical_errors"), ensure_ascii=False
                    ),
                    label + " critique": judgment.get("critique", "Not semantically assessed"),
                }
            )
        combined.append(row)
    with (args.output / "case-reviews.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(
        json.dumps(
            {
                arm["run"] + "/" + arm["arm"]: Counter(arm["metrics"]["pooled"]["counts"])
                for arm in arms
            }
        )
    )


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY in the process environment")
    main()
