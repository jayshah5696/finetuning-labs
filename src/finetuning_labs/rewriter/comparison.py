"""Recompute a paired diagnostic report from exact saved candidates and judgments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .benchmark import Bundle, judge_packet, load_bundle
from .evaluation import _load_generations, aggregate, assess, bundle_identity, digest, write_json


def _record(path: Path) -> dict:
    record = json.loads(path.read_text())
    if digest(record["request"]) != record["request_sha256"]:
        raise ValueError("Saved request hash differs from its contents")
    if record["status"] == "completed" and digest(record["result"]) != record["result_sha256"]:
        raise ValueError("Saved response hash differs from its contents")
    return record


def _bound_assessments(bundle: Bundle, generation_dir: Path, judge_dir: Path):
    generations = _load_generations(bundle, generation_dir)
    spec = json.loads((judge_dir / "run.json").read_text())["spec"]
    if spec["bundle_sha256"] != bundle_identity(bundle):
        raise ValueError("Judge run uses a different bundle")
    if spec["generation_sha256"] != digest(generations):
        raise ValueError("Judge run is not bound to these candidates")
    rows, records = [], []
    for case, generation in zip(bundle.cases, generations, strict=True):
        judgment, error = None, False
        if generation["status"] == "completed":
            record = _record(judge_dir / "calls" / f"{case.id}.json")
            expected = {"packet": judge_packet(case, generation["text"]), "config": spec["config"]}
            if record["request"] != expected:
                raise ValueError("Judge packet does not match the exact candidate and case")
            records.append(record)
            if record["status"] == "completed":
                judgment = record["result"].get("judgment")
                error = judgment is None
            else:
                error = True
        rows.append(assess(case, generation, judgment, judge_error=error))
    return rows, records, spec["config"]


def _usage(records: list[dict]) -> dict:
    usages = [
        r["result"]["usage"]
        for r in records
        if r["status"] == "completed" and isinstance(r["result"].get("usage"), dict)
    ]
    gateway = [u["cost"] for u in usages if isinstance(u.get("cost"), (int, float))]
    upstream = [(u.get("cost_details") or {}).get("upstream_inference_cost") for u in usages]
    upstream = [cost for cost in upstream if isinstance(cost, (int, float))]
    return {
        "calls": len(records),
        "usage_records": len(usages),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usages) if usages else None,
        "output_tokens": sum(u.get("output_tokens", 0) for u in usages) if usages else None,
        "gateway_reported_cost_usd": sum(gateway) if gateway else None,
        "upstream_reported_cost_usd": sum(upstream) if upstream else None,
        "upstream_cost_coverage": len(upstream),
        "byok_calls": sum(u.get("is_byok") is True for u in usages),
    }


def paired_report(
    bundle: Bundle,
    left_generation: Path,
    right_generation: Path,
    left_judge: Path,
    right_judge: Path,
) -> dict:
    left_spec = json.loads((left_generation / "run.json").read_text())["spec"]
    right_spec = json.loads((right_generation / "run.json").read_text())["spec"]
    if left_spec["prompt_sha256"] != right_spec["prompt_sha256"]:
        raise ValueError("Generator behavioral prompts differ")
    left_rows, left_calls, left_config = _bound_assessments(bundle, left_generation, left_judge)
    right_rows, right_calls, right_config = _bound_assessments(
        bundle, right_generation, right_judge
    )
    if left_config != right_config:
        raise ValueError("Both arms must use the same judge settings and rubric")
    pairs = Counter()
    for left, right in zip(left_rows, right_rows, strict=True):
        lpass, rpass = left["status"] == "pass", right["status"] == "pass"
        pairs[
            "both_pass"
            if lpass and rpass
            else "right_only_pass"
            if rpass
            else "left_only_pass"
            if lpass
            else "neither_verified_pass"
        ] += 1
    n = len(bundle.cases)
    gen_calls = {
        "left": [_record(left_generation / "calls" / f"{c.id}.json") for c in bundle.cases],
        "right": [_record(right_generation / "calls" / f"{c.id}.json") for c in bundle.cases],
    }
    return {
        "purpose": "Sol-assessed diagnostic comparison; human calibration is pending",
        "human_calibrated": False,
        "locked_test": False,
        "scheduled_pairs": n,
        "bundle_sha256": bundle_identity(bundle),
        "judge_config": left_config,
        "judge_rubric_sha256": digest(left_config.get("rubric", "")),
        "left_model": left_spec["config"]["model"],
        "right_model": right_spec["config"]["model"],
        "left": aggregate(bundle, left_rows),
        "right": aggregate(bundle, right_rows),
        "pairs": {
            name: pairs[name]
            for name in ["both_pass", "right_only_pass", "left_only_pass", "neither_verified_pass"]
        },
        "pooled_gain_percentage_points": 100
        * (pairs["right_only_pass"] - pairs["left_only_pass"])
        / n
        if n
        else None,
        "api_usage": {
            "left_generation": _usage(gen_calls["left"]),
            "right_generation": _usage(gen_calls["right"]),
            "left_judge": _usage(left_calls),
            "right_judge": _usage(right_calls),
        },
        "assessments": {"left": left_rows, "right": right_rows},
    }


def markdown(report: dict) -> str:
    def rate(value):
        return "N/A" if value is None else f"{value:.1f}%"

    lines = [
        "# Liquid and Luna diagnostic comparison",
        "",
        "Sol assessed both models with the same rubric and hidden generator identity.",
        "These are diagnostic scores on 68 calibration cases. Human judge calibration and",
        "an independent test set are pending. This is not a release-quality accuracy claim.",
        "",
        f"Left: `{report['left_model']}`. Right: `{report['right_model']}`.",
        f"Judge: `{report['judge_config']['model']}` through OpenRouter and Arka.",
        "",
        "| Slice | Stock Liquid passes | Luna passes |",
        "| --- | ---: | ---: |",
    ]
    for mode, left in report["left"]["per_mode"].items():
        right = report["right"]["per_mode"][mode]
        lines.append(
            f"| {mode} | {left['counts']['pass']}/{left['scheduled']} | "
            f"{right['counts']['pass']}/{right['scheduled']} |"
        )
    left, right = report["left"]["pooled"], report["right"]["pooled"]
    lines += [
        f"| Pooled | {left['counts']['pass']}/{left['scheduled']} | {right['counts']['pass']}/{right['scheduled']} |",
        f"| Equal-mode headline | {rate(report['left']['headline_equal_mode_pass_rate_pct'])} | {rate(report['right']['headline_equal_mode_pass_rate_pct'])} |",
    ]
    for length in ["short", "long"]:
        ls, rs = report["left"]["by_length"][length], report["right"]["by_length"][length]
        lines.append(
            f"| {length.capitalize()} | {ls['counts']['pass']}/{ls['scheduled']} | {rs['counts']['pass']}/{rs['scheduled']} |"
        )
    lines += [
        "",
        "| Diagnostic | Stock Liquid | Luna |",
        "| --- | ---: | ---: |",
        f"| Definite assessment coverage | {rate(left['assessment_coverage_pct'])} | {rate(right['assessment_coverage_pct'])} |",
        f"| Semantic review coverage | {left['semantic_reviewed']}/{left['scheduled']} | {right['semantic_reviewed']}/{right['scheduled']} |",
        f"| Outputs with a judge-flagged critical error | {left['critical_error_count']} | {right['critical_error_count']} |",
        f"| Explicit identity preserved | {left['identity_unchanged']}/{left['identity_count']} | {right['identity_unchanged']}/{right['identity_count']} |",
        f"| Judge errors | {left['judge_error_events']} | {right['judge_error_events']} |",
        f"| Truncations | {left['counts']['truncated']} | {right['counts']['truncated']} |",
        f"| Uncertain | {left['counts']['uncertain']} | {right['counts']['uncertain']} |",
        "",
        "Every scheduled case stays in the pass-rate denominator. A pass requires complete",
        "generation, declared mechanical checks, and all five semantic criteria. Critical",
        "errors cannot be offset by style. A missing judgment never becomes a pass.",
        "",
        "Paired outcomes: " + ", ".join(f"{name}={n}" for name, n in report["pairs"].items()) + ".",
        f"Luna's pooled difference is {report['pooled_gain_percentage_points']:.1f} percentage points.",
        "",
        "The eight long cases represent four source documents. The three identity cases are",
        "too few for a precise generalization claim. All references remain draft.",
        "",
        "API usage is retained in comparison.json. OpenRouter reported BYOK calls. A zero",
        "gateway charge does not mean free inference; upstream provider-reported costs are",
        "recorded separately. GPU cost is not included in those API cost fields.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = paired_report(
        load_bundle(),
        args.run / "liquid",
        args.run / "luna",
        args.run / "liquid-judge",
        args.run / "luna-judge",
    )
    output = args.output or args.run / "comparison"
    write_json(output / "comparison.json", report)
    (output / "REPORT.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
