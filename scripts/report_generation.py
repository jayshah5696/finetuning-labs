"""Report only what a collected generation run proves before semantic judging."""

import argparse
from collections import Counter
from pathlib import Path

from finetuning_labs.rewriter.benchmark import MODES, deterministic_checks, load_bundle
from finetuning_labs.rewriter.evaluation import _load_generations, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_bundle()
    generations = _load_generations(bundle, args.generation)
    long_ids = set(bundle.manifest.get("long_form", {}).get("case_ids", []))
    rows = []
    for case, generation in zip(bundle.cases, generations, strict=True):
        failures = deterministic_checks(case, generation["text"])
        rows.append(
            {
                "case_id": case.id,
                "mode": case.mode,
                "generation_status": generation["status"],
                "mechanical_pass": generation["status"] == "completed" and not failures,
                "mechanical_failures": failures,
                "long": case.id in long_ids,
                "identity_required": case.rules.identity,
                "identity_unchanged": generation["status"] == "completed"
                and generation["text"] == case.input,
                "output_tokens": generation.get("output_tokens"),
            }
        )

    def counts(selected):
        return {
            "scheduled": len(selected),
            "mechanical_passes": sum(r["mechanical_pass"] for r in selected),
            "execution": dict(Counter(r["generation_status"] for r in selected)),
        }

    identity = [r for r in rows if r["identity_required"]]
    report = {
        "claim": "Declared mechanical checks only; not semantic accuracy",
        "semantic_judge_status": "not run",
        "quality_pass_rate_pct": None,
        "human_calibrated": False,
        "locked_test": False,
        "pooled": counts(rows),
        "per_mode": {mode: counts([r for r in rows if r["mode"] == mode]) for mode in MODES},
        "by_length": {
            "long": counts([r for r in rows if r["long"]]),
            "short": counts([r for r in rows if not r["long"]]),
        },
        "identity": {
            "scheduled": len(identity),
            "unchanged": sum(r["identity_unchanged"] for r in identity),
        },
        "runtime": generations[0].get("runtime"),
        "cases": rows,
    }
    write_json(args.output / "mechanical.json", report)
    lines = [
        "# Stock Liquid diagnostic baseline",
        "",
        "This report measures declared mechanical checks. It is not a semantic quality score.",
        "Human calibration and the Sol judge run remain pending.",
        "",
        "Model: `LiquidAI/LFM2.5-230M` at `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.",
        "Runtime: NVIDIA L4 on Modal, PyTorch 2.13.0, BF16, eager attention, greedy decoding.",
        "The reference convolution path was used because causal_conv1d was not installed.",
        "This is a functional baseline, not an optimized latency benchmark.",
        "",
        "| Mode | Scheduled | Mechanical passes |",
        "| --- | ---: | ---: |",
    ]
    for mode, stats in report["per_mode"].items():
        lines.append(f"| {mode} | {stats['scheduled']} | {stats['mechanical_passes']} |")
    total = report["pooled"]
    long = report["by_length"]["long"]
    lines += [
        "",
        f"Total: {total['mechanical_passes']}/{total['scheduled']} mechanical passes.",
        f"Generation outcomes: {total['execution']}.",
        f"Long cases: {long['mechanical_passes']}/{long['scheduled']} mechanical passes.",
        f"Explicit identity cases: {report['identity']['unchanged']}/{len(identity)} unchanged.",
        "",
        "Every successful mechanical result still needs semantic review. A normal end token",
        "does not prove that every paragraph survived. The long outputs include missing quoted",
        "content, shortened documents, and added commentary. Preserve the raw outputs for review.",
        "",
        "Luna and Sol are not reported as evaluated without actual provider calls.",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:25]))


if __name__ == "__main__":
    main()
