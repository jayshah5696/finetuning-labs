"""Native Arka generation and decision stages for independently reviewed rewrite data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .benchmark import MODES, Case, Rules
from .evaluation import digest, write_json
from .source_briefs import scenarios
from .training import Example, _tokenizer, token_audit, validation_index

ARKA_PIN = "5d353740ae1dd34742841071752814028db1489a"
GENERATION_PROMPT = """Create ONE fictional rewrite-training pair using the scenario below.
The scenario is authoring data. Use its facts and requested genre as constraints.
Return the structured instruction and response fields only, with no reasoning.
The instruction must start with the scenario's exact header, followed by the raw text to rewrite.
The response must contain only the cleaned text, without that header.
Do not write an answer to the raw text's request. In /prompt the response must itself be a prompt.

Follow the mode_contract in the scenario. Preserve facts, requests versus assertions, negation,
conditions, uncertainty, quotes, names, and literals. Do not add outcomes, approvals or conclusions.
For transcribe, prefer missing punctuation, mostly lowercase speech, genuine restarts, unambiguous
self-corrections, spoken list cues and dictionary spellings. Preserve meaningful filler-like words.
Do not paraphrase already clear words in transcribe. For slack, keep casual tone and contractions.
For polish, improve genuinely awkward wording. For prompt, organize a realistic rough request.

Use natural imperfect input, not artificial corruption. Never delete all articles, double spaces
as the only error, or sprinkle fillers onto an otherwise fully edited essay. Most nonidentity
English input should start lowercase. Do not make every pair a grammar-only miniature.
Identity scenarios must have exactly equal input and response and already be suitable for the mode.

Short means roughly 20 to 100 words. Medium means roughly 170 to 330 words.
Long means 1000 to 1800 Liquid tokenizer tokens of INPUT: aim for 1000 to 1300 substantive words,
with varied concrete facts in every paragraph. Do not pad four facts into an essay. Long slack
is a pasted thread of short messages, not one giant Slack post. Preserve speaker labels.
Long prompt is a rough brief to organize, never the work requested by that brief.
Long transcribe should look like actual unpunctuated dictation, not an edited essay plus 'um'.
The input and target must both be complete. No truncation or missing final section.

Keep authoring instructions OUT of the raw text. Never write 'as data', 'do not invent a result',
'canonical', 'formatter', or commentary about preserving a quote unless the actual scenario
requires that exact content. Quotes should be ordinary reported speech, not repeated injection tests.
Use the scenario language throughout except supplied proper names and dictionary literals.

Scenario:
{seed_instruction}
"""

DIMENSIONS = {
    "candidate_input": "Assess only the text after INPUT in instruction. Is it a realistic recoverable user draft, dictation, or pasted thread? Imperfect punctuation and grammar are intentional. Reject fake word deletion, padding, meta authoring rules or ambiguity with no recoverable final intent.",
    "output_quality": "Assess only response as a rewritten text in the supplied MODE. Is it readable and natural for that mode? Preserve intentional unfinished thoughts. Slack stays casual; long slack may be a thread, never force a formal essay. Do not evaluate whether it fulfills an underlying prompt.",
    "pair_facts": "Does response preserve all substantive facts, sections, conditions, negation and uncertainty in the INPUT portion of instruction? No invented facts, missing final sections, approvals or summary. Identical input and response preserve meaning by definition.",
    "pair_intent": "Does a question remain a question, a request remain a request, and a proposal remain unapproved? Explicit self-corrections should follow final intent. In /prompt the response is itself the reorganized request, NOT an answer or completed task.",
    "pair_literals": "Are names, quoted wording, quantities with their roles, URLs, code and filenames preserved, allowing explicitly supplied dictionary normalization and explicit spoken corrections? Obsolete corrected details should not be forced into the target.",
    "pair_mode": "Follow the mode in instruction. /slack minimal casual cleanup. /transcribe clean actual disfluencies and punctuation without lexical paraphrasing. /prompt organize request only. /polish improve awkward prose without changing scope or certainty. No added wrapper or formatter commentary. Already correct text may remain unchanged.",
}


def header(plan):
    return Case(
        id="header",
        input="",
        output="",
        mode=plan["mode"],
        checks="",
        context=plan["context"],
        language=plan["language"],
    ).user_message()


def generation_config(seed_path: str, count: int, *, workers=4):
    return {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "openai/gpt-6-luna",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 180,
            "max_retries": 1,
            "supports_json_schema": True,
        },
        "decisions": {
            "provider": "openrouter",
            "model": "openai/gpt-6-luna-decisions",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/alpha",
            "max_state_chars": 100000,
            "timeout_seconds": 60,
            "max_retries": 1,
        },
        "executor": {"mode": "threadpool", "max_workers": workers},
        "pipeline": [
            {"type": "seed_source", "path": seed_path},
            {"type": "normalize_conversation"},
            {
                "type": "prompt_based_generator",
                "target_count": count,
                "generation_multiplier": 1,
                "prompt_template": GENERATION_PROMPT,
                "temperature": 0.6,
                "max_tokens": 8000,
            },
            {"type": "exact"},
            {
                "type": "length",
                "min_instruction_chars": 1,
                "max_instruction_chars": 24000,
                "min_response_chars": 1,
                "max_response_chars": 24000,
            },
            {"type": "decision_panel", "include": ["labeling_engine"]},
            {
                "type": "labeling_engine",
                "backend": "decision",
                "rubric_path": "rubric.yaml",
                "min_overall_score": 0.80,
                "min_confidence": 0.20,
                "on_uncertain": "escalate",
            },
        ],
        "output": {
            "format": "jsonl",
            "path": "native-output.jsonl",
            "on_unresolved_escalation": "drop",
        },
    }


def rubric():
    return {
        "version": "2.0",
        "description": "Input, output and pair quality for a text formatter",
        "dimensions": [
            {
                "name": name,
                "description": text,
                "scale_min": 0,
                "scale_max": 1,
                "criteria": {
                    0: "Contains a concrete defect for this criterion",
                    1: "Meets this criterion without a concrete defect",
                },
            }
            for name, text in DIMENSIONS.items()
        ],
        "overall_weights": dict.fromkeys(DIMENSIONS, 1 / len(DIMENSIONS)),
        "few_shot": [],
    }


def parse_candidate(plan, instruction, response):
    expected = header(plan)
    if not instruction.startswith(expected):
        raise ValueError("generated mode, language or context header differs from source plan")
    raw = instruction[len(expected) :]
    if not raw.strip() or not response.strip():
        raise ValueError("empty candidate or output")
    return {
        "input": raw,
        "output": response,
        "mode": plan["mode"],
        "checks": MODES[plan["mode"]]
        + " Preserve every fact, request, condition and quoted literal.",
    }


def candidate_issues(plan, row):
    issues = []
    raw, target = row["input"], row["output"]
    if (
        not plan.get("identity")
        and re.sub(r"\s+", " ", raw).strip() == re.sub(r"\s+", " ", target).strip()
    ):
        issues.append("whitespace_only")
    if plan.get("identity") and raw != target:
        issues.append("identity_changed")
    if re.search(
        r"as data|do not invent a result|canonical|text formatter",
        raw + "\n" + target,
        re.IGNORECASE,
    ):
        issues.append("authoring_meta_leak")
    sentences = [x.strip().casefold() for x in re.split(r"[.!?]\s+", target) if len(x.strip()) > 50]
    if len(sentences) - len(set(sentences)) >= 2:
        issues.append("repeated_sentences")
    return issues


def release_eligible(row_sha, review, mechanical_pass):
    return bool(
        mechanical_pass
        and review
        and review.get("row_sha256") == row_sha
        and review.get("verdict") == "pass"
    )


def write_verification_pack(rows, metadata, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        items = []
        for row, meta in zip(rows, metadata, strict=True):
            if row["mode"] == mode:
                items.append(
                    {
                        "id": meta["id"],
                        "row_sha256": digest(row),
                        "mode": mode,
                        "language": meta["language"],
                        "context": meta["context"],
                        "mode_instructions": MODES[mode],
                        "input": row["input"],
                        "output": row["output"],
                    }
                )
        (destination / f"{mode[1:]}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
        )


def verify_existing_validation(root: Path):
    import yaml
    from arka.config.loader import ConfigLoader
    from arka.pipeline.runner import PipelineRunner
    from arka.pipeline.stage_builder import StageBuilder

    from .benchmark import load_bundle

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    rows = [
        {k: getattr(case, k) for k in ["input", "output", "mode", "checks"]}
        for case in bundle.cases
    ]
    metadata = [
        {"id": case.id, "language": case.language, "context": case.context} for case in bundle.cases
    ]
    seeds = [{"instruction": case.user_message(), "response": case.output} for case in bundle.cases]
    (root / "seeds.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in seeds)
    )
    (root / "rubric.yaml").write_text(yaml.safe_dump(rubric(), sort_keys=False))
    config = generation_config("seeds.jsonl", len(seeds))
    config["pipeline"] = [x for x in config["pipeline"] if x["type"] != "prompt_based_generator"]
    (root / "pipeline.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    resolved = ConfigLoader().load(root / "pipeline.yaml")
    result = PipelineRunner(root).run(resolved, StageBuilder(resolved, root).build(), "validation")
    write_verification_pack(rows, metadata, root / "opus-inputs")
    return {
        "validation_cases": len(rows),
        "native_output_count": result.final_count,
        "original_benchmark_modified": False,
        "opus_verified": False,
    }


def rescore_candidates(source_root: Path, root: Path):
    """Re-check every token-valid candidate through the native Arka decision pipeline."""
    import yaml
    from arka.config.loader import ConfigLoader
    from arka.pipeline.output import OutputWriter
    from arka.pipeline.runner import PipelineRunner
    from arka.pipeline.stage_builder import StageBuilder

    rows = [json.loads(x) for x in (source_root / "candidates.jsonl").read_text().splitlines()]
    metadata = json.loads((source_root / "supporting.json").read_text())
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = [
        {
            "instruction": header({**meta, "mode": row["mode"]}) + row["input"],
            "response": row["output"],
        }
        for row, meta in zip(rows, metadata, strict=True)
    ]
    if any(len(value) > 16384 for seed in seeds for value in seed.values()):
        raise ValueError("candidate exceeds Arka source character cap; refusing silent truncation")
    (root / "seeds.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in seeds)
    )
    (root / "rubric.yaml").write_text(yaml.safe_dump(rubric(), sort_keys=False))
    config = generation_config("seeds.jsonl", len(seeds))
    config["pipeline"] = [x for x in config["pipeline"] if x["type"] != "prompt_based_generator"]
    (root / "pipeline.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    resolved = ConfigLoader().load(root / "pipeline.yaml")
    result = PipelineRunner(root).run(resolved, StageBuilder(resolved, root).build(), "checks")
    writer = OutputWriter()
    stages = root / "runs/checks/stages"
    sources = writer.read_parquet(stages / "01_source/data.parquet")
    mapping = {r.id: meta for r, meta in zip(sources, metadata, strict=True)}
    scored = []
    for name in ["data.parquet", "dropped.parquet"]:
        path = stages / "03_label_quality" / name
        if path.exists():
            scored.extend(writer.read_parquet(path))
    decisions = {}
    for record in scored:
        meta = mapping[record.id]
        decisions[meta["id"]] = {
            k: v.model_dump(mode="json") for k, v in record.scores.decisions.items()
        }
    for meta in metadata:
        meta["decisions"] = decisions.get(meta["id"], {})
        meta["mechanical_issues"] = [
            x for x in meta["mechanical_issues"] if x != "missing_pipeline_decisions"
        ]
        if not meta["decisions"]:
            meta["mechanical_issues"].append("missing_pipeline_decisions")
    write_json(source_root / "scored-supporting.json", metadata)
    for split in ["train", "dev", "validation"]:
        selected = [(r, m) for r, m in zip(rows, metadata, strict=True) if m["split"] == split]
        write_verification_pack(
            [x[0] for x in selected], [x[1] for x in selected], source_root / f"opus-{split}"
        )
    return {
        "candidates": len(rows),
        "decision_scored": len(decisions),
        "native_output_count": result.final_count,
        "arka_revision": ARKA_PIN,
    }


def run_generation(
    root: Path,
    *,
    limit=None,
    resume=False,
    plans_override=None,
    prompt_override=None,
    workers=4,
    rubric_override=None,
):
    import yaml
    from arka.config.loader import ConfigLoader
    from arka.pipeline.output import OutputWriter
    from arka.pipeline.runner import PipelineRunner
    from arka.pipeline.stage_builder import StageBuilder

    root = root.resolve()
    plans = plans_override if plans_override is not None else scenarios()
    prompt = prompt_override or GENERATION_PROMPT
    quality_rubric = rubric_override or rubric()
    if limit:
        plans = [x for mode in MODES for x in [p for p in plans if p["mode"] == mode][:limit]]
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "arka_revision": ARKA_PIN,
        "plans": plans,
        "prompt_sha256": digest(prompt),
        "rubric_sha256": digest(quality_rubric),
    }
    if (root / "source-plan.json").exists():
        if json.loads((root / "source-plan.json").read_text()) != spec:
            raise ValueError("source plan changed; use a new run directory")
    else:
        write_json(root / "source-plan.json", spec)
        seeds = [
            {
                "instruction": json.dumps(
                    {**plan, "header": header(plan), "mode_contract": MODES[plan["mode"]]},
                    ensure_ascii=False,
                ),
                "response": "",
            }
            for plan in plans
        ]
        (root / "seeds.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in seeds)
        )
        (root / "rubric.yaml").write_text(
            yaml.safe_dump(quality_rubric, sort_keys=False, allow_unicode=True)
        )
        config = generation_config("seeds.jsonl", len(plans), workers=workers)
        next(x for x in config["pipeline"] if x["type"] == "prompt_based_generator")[
            "prompt_template"
        ] = prompt
        (root / "pipeline.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    cfg = ConfigLoader().load(root / "pipeline.yaml")
    stages = StageBuilder(cfg, root).build()
    result = PipelineRunner(root).run(cfg, stages, "author", resume=resume)
    writer = OutputWriter()
    run_dir = root / "runs/author/stages"
    sources = writer.read_parquet(run_dir / "01_source/data.parquet")
    mapping = {r.id: plan for r, plan in zip(sources, plans, strict=True)}
    generated = writer.read_parquet(run_dir / "02_generate/data.parquet")
    scored = {}
    exported = {
        digest(json.loads(line)) for line in (root / "native-output.jsonl").read_text().splitlines()
    }
    for name in ["data.parquet", "dropped.parquet"]:
        path = run_dir / "03_label_quality" / name
        if path.exists():
            records = writer.read_parquet(path)
            scored.update({r.id: r for r in records})
    tokenizer, blocked = _tokenizer(), validation_index()
    rows, metadata, rejected = [], [], []
    for record in generated:
        plan = mapping[record.lineage.root_id]
        try:
            row = parse_candidate(plan, record.payload.instruction, record.payload.response)
            issues = candidate_issues(plan, row)
            case = Case(
                id=plan["id"],
                **row,
                context=plan["context"],
                language=plan["language"],
                rules=Rules(identity=plan["identity"]),
            )
            tokens = token_audit(
                Example(
                    case,
                    "train" if plan["split"] == "train" else "validation",
                    plan["source_group"],
                    plan["length"],
                    "independent",
                    plan["id"],
                ),
                tokenizer,
            )
            if blocked.matches(row["input"]) or blocked.matches(row["output"]):
                issues.append("heldout_overlap")
            decision_record = scored.get(record.id)
            decisions = (
                {k: v.model_dump(mode="json") for k, v in decision_record.scores.decisions.items()}
                if decision_record
                else {}
            )
            if not decisions:
                issues.append("missing_pipeline_decisions")
            rows.append(row)
            metadata.append(
                {
                    **plan,
                    "row_sha256": digest(row),
                    "record_id": record.id,
                    "mechanical_issues": issues,
                    "decisions": decisions,
                    "native_accepted": digest(record.export_payload()) in exported,
                    "tokens": {k: v for k, v in tokens.items() if k not in {"input_ids", "labels"}},
                    "opus_review": "pending",
                }
            )
        except ValueError as error:
            rejected.append({"id": plan["id"], "record_id": record.id, "reason": str(error)})
    (root / "candidates.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    )
    write_json(root / "supporting.json", metadata)
    write_json(root / "rejected.json", rejected)
    for split in ["train", "dev", "validation"]:
        selected = [
            (row, meta) for row, meta in zip(rows, metadata, strict=True) if meta["split"] == split
        ]
        write_verification_pack(
            [x[0] for x in selected], [x[1] for x in selected], root / f"opus-{split}"
        )
    return {
        "planned": len(plans),
        "generated": len(generated),
        "token_valid_candidates": len(rows),
        "mechanically_clean": sum(not x["mechanical_issues"] for x in metadata),
        "rejected": len(rejected),
        "native_output_count": result.final_count,
        "opus_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Number of briefs per mode for a pilot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_generation(args.run, limit=args.limit, resume=args.resume)
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
