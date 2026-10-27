"""Durable diagnostic baselines. Human calibration remains a separate claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import (
    COMMON_SYSTEM,
    MODES,
    Bundle,
    Case,
    deterministic_checks,
    judge_packet,
    load_bundle,
    validate_judgment,
)

ARKA_REVISION = "779040204b0f392465b7e6240bf9d9200816dc5d"
STATUSES = ("pass", "fail", "uncertain", "judge_error", "generation_error", "truncated", "pending")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def saved_call(path: Path, request: dict, invoke: Callable[[], dict]) -> dict:
    """Never reissue an existing call; a changed or corrupted request is an error."""
    if path.exists():
        record = json.loads(path.read_text())
        if record["request_sha256"] != digest(request) or digest(record["request"]) != digest(
            request
        ):
            raise ValueError("saved request differs from the requested call")
        if record["status"] == "completed" and digest(record["result"]) != record["result_sha256"]:
            raise ValueError("saved result hash differs from its contents")
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "started",
        "request": request,
        "request_sha256": digest(request),
        "started_at": datetime.now(UTC).isoformat(),
    }
    # Reserve before spending. An interrupted started record is never an implicit retry.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    started = time.perf_counter()
    try:
        result = invoke()
        record.update(status="completed", result=result, result_sha256=digest(result))
    except Exception as error:  # noqa: BLE001 -- preserve a failed call without exposing its payload
        # Provider messages can include credentials. Keep only a safe error type and status.
        record.update(status="error", error_type=type(error).__name__)
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            record["http_status"] = status
    record["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    write_json(path, record)
    return record


def _run_manifest(path: Path, spec: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["spec"] != spec:
            raise ValueError("run identity changed; use a new output directory")
    else:
        write_json(manifest, {"spec": spec, "created_at": datetime.now(UTC).isoformat()})


def bundle_identity(bundle: Bundle) -> str:
    return digest([asdict(case) for case in bundle.cases])


def _generation(case: Case, record: dict) -> dict:
    if record["status"] != "completed":
        return {
            "case_id": case.id,
            "status": "generation_error",
            "text": "",
            "error_type": record.get("error_type", record["status"]),
        }
    result = record["result"]
    text = result.get("text")
    reason = result.get("finish_reason")
    if result.get("status") == "context_overflow":
        status = "context_overflow"
    elif reason in {"length", "max_output_tokens"}:
        status = "truncated"
    elif not isinstance(text, str) or not text.strip() or reason not in {"stop", "eos"}:
        status = "generation_error"
    else:
        status = "completed"
    return {**result, "case_id": case.id, "status": status, "text": text or ""}


def generate_cases(
    bundle: Bundle,
    destination: Path,
    config: dict,
    generate: Callable[[list[dict]], dict],
    *,
    workers: int = 1,
) -> list[dict]:
    spec = {
        "stage": "generation",
        "bundle_sha256": bundle_identity(bundle),
        "config": config,
        "case_ids": [case.id for case in bundle.cases],
        "prompt_sha256": digest({"system": COMMON_SYSTEM, "modes": MODES}),
    }
    _run_manifest(destination, spec)

    def one(case):
        messages = [
            {"role": "system", "content": COMMON_SYSTEM + MODES[case.mode]},
            {"role": "user", "content": case.user_message()},
        ]
        record = saved_call(
            destination / "calls" / f"{case.id}.json",
            {"case_id": case.id, "config": config, "messages": messages},
            lambda: generate(messages),
        )
        result = _generation(case, record)
        print(f"generation {case.id}: {result['status']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, bundle.cases))
    write_json(destination / "generations.json", results)
    return results


def _load_generations(bundle: Bundle, destination: Path) -> list[dict]:
    manifest = json.loads((destination / "run.json").read_text())
    if manifest["spec"]["bundle_sha256"] != bundle_identity(bundle):
        raise ValueError("generation run belongs to a different case bundle")
    result = []
    for case in bundle.cases:
        record = json.loads((destination / "calls" / f"{case.id}.json").read_text())
        if digest(record["request"]) != record["request_sha256"]:
            raise ValueError("saved request hash differs from its contents")
        if record["status"] == "completed" and digest(record["result"]) != record["result_sha256"]:
            raise ValueError("saved generation hash differs from its contents")
        result.append(_generation(case, record))
    return result


def judge_cases(
    bundle: Bundle,
    generation_dir: Path,
    destination: Path,
    config: dict,
    judge: Callable[[dict], dict],
    *,
    workers: int = 1,
) -> list[dict]:
    generations = _load_generations(bundle, generation_dir)
    spec = {
        "stage": "judge",
        "bundle_sha256": bundle_identity(bundle),
        "config": config,
        "generation_sha256": digest(generations),
    }
    _run_manifest(destination, spec)

    def one(pair):
        case, generation = pair
        judgment = None
        error = False
        if generation["status"] == "completed":
            packet = judge_packet(case, generation["text"])
            record = saved_call(
                destination / "calls" / f"{case.id}.json",
                {"packet": packet, "config": config},
                lambda: judge(packet),
            )
            if record["status"] == "completed":
                judgment = record["result"].get("judgment")
                error = judgment is None
            else:
                error = True
        result = assess(case, generation, judgment, judge_error=error)
        print(f"judge {case.id}: {result['status']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        assessments = list(pool.map(one, zip(bundle.cases, generations, strict=True)))
    write_json(destination / "assessments.json", assessments)
    write_json(destination / "metrics.json", aggregate(bundle, assessments))
    return assessments


def assess(case: Case, generation: dict, judgment: dict | None, *, judge_error=False) -> dict:
    if generation["case_id"] != case.id:
        raise ValueError("generation case does not match the case")
    text = generation.get("text", "")
    result = {
        "case_id": case.id,
        "status": "pending",
        "critical_error": False,
        "semantic_reviewed": False,
        "identity_required": case.rules.identity,
        "identity_unchanged": False,
        "mechanical_failures": [],
        "judge_error": judge_error,
    }
    status = generation["status"]
    if status != "completed" or not isinstance(text, str) or not text.strip():
        result["status"] = "truncated" if status == "truncated" else "generation_error"
        return result
    result["identity_unchanged"] = text == case.input
    failures = deterministic_checks(case, text)
    result["mechanical_failures"] = failures
    if judgment is not None:
        try:
            validate_judgment(case.id, judgment)
        except (ValueError, TypeError, KeyError):
            result["judge_error"] = True
        else:
            result.update(
                semantic_reviewed=True,
                critical_error=bool(judgment["critical_errors"]),
                judgment=judgment,
            )
    result["status"] = (
        "fail"
        if failures
        else "judge_error"
        if result["judge_error"]
        else judgment["verdict"]
        if result["semantic_reviewed"]
        else "pending"
    )
    return result


def aggregate(bundle: Bundle, assessments: list[dict]) -> dict:
    if Counter(row["case_id"] for row in assessments) != Counter(case.id for case in bundle.cases):
        raise ValueError("assessments must contain every scheduled case exactly once")
    by_id = {case.id: case for case in bundle.cases}

    def stats(rows):
        n = len(rows)
        counts = Counter(row["status"] for row in rows)
        if set(counts) - set(STATUSES):
            raise ValueError("unknown assessment status")
        p, f = counts["pass"], counts["fail"]
        identities = [row for row in rows if row["identity_required"]]

        def pct(value, denominator):
            return 100 * value / denominator if denominator else None

        return {
            "scheduled": n,
            "counts": {s: counts[s] for s in STATUSES},
            "verified_pass_rate_pct": pct(p, n),
            "decided_case_pass_rate_pct": pct(p, p + f),
            "assessment_coverage_pct": pct(p + f, n),
            "critical_error_count": sum(row["critical_error"] for row in rows),
            "critical_error_rate_pct": pct(sum(row["critical_error"] for row in rows), n),
            "semantic_reviewed": sum(row["semantic_reviewed"] for row in rows),
            "semantic_review_coverage_pct": pct(sum(row["semantic_reviewed"] for row in rows), n),
            "identity_count": len(identities),
            "identity_unchanged": sum(row["identity_unchanged"] for row in identities),
            "identity_rate_pct": pct(
                sum(row["identity_unchanged"] for row in identities), len(identities)
            ),
            "judge_error_events": sum(row.get("judge_error", False) for row in rows),
        }

    per_mode = {
        mode: stats([row for row in assessments if by_id[row["case_id"]].mode == mode])
        for mode in MODES
        if any(case.mode == mode for case in bundle.cases)
    }
    long_ids = set(bundle.manifest.get("long_form", {}).get("case_ids", []))
    return {
        "purpose": "diagnostic calibration; judge is not human-calibrated",
        "human_calibrated": False,
        "locked_test": False,
        "bundle_sha256": bundle_identity(bundle),
        "pooled": stats(assessments),
        "headline_equal_mode_pass_rate_pct": sum(
            mode["verified_pass_rate_pct"] for mode in per_mode.values()
        )
        / len(per_mode)
        if per_mode
        else None,
        "per_mode": per_mode,
        "by_length": {
            "long": stats([r for r in assessments if r["case_id"] in long_ids]),
            "short": stats([r for r in assessments if r["case_id"] not in long_ids]),
        },
    }


def liquid_backend(config: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The Liquid evaluation requires the selected CUDA L4 runtime")
    device = torch.cuda.get_device_name(0)
    if "L4" not in device:
        raise RuntimeError(f"Expected L4, received {device}")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config["revision"],
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    runtime = {
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "dtype": "bfloat16",
        "attention": "eager",
        "seed": 42,
    }

    def generate(messages):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        prompt_count = encoded["input_ids"].shape[-1]
        if prompt_count + config["max_new_tokens"] > config["max_seq_length"]:
            return {"status": "context_overflow", "text": "", "prompt_tokens": prompt_count}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        tokens = output[0, prompt_count:].tolist()
        eos = tokenizer.eos_token_id
        complete = bool(tokens) and tokens[-1] == eos
        return {
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "finish_reason": "eos" if complete else "length",
            "model": config["model"],
            "revision": config["revision"],
            "prompt_tokens": prompt_count,
            "output_tokens": len(tokens),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "runtime": runtime,
        }

    return generate


def hosted_backend(config: dict, *, judge=False):
    from arka.config.models import LLMConfig
    from arka.llm.openai_client import build_openai_client

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the requested Luna and Sol calls")
    client = build_openai_client(
        LLMConfig(
            provider="openai",
            model=config["model"],
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            timeout_seconds=180,
            max_retries=1,
        )
    )

    def call(value):
        messages = (
            [
                {"role": "system", "content": config["rubric"]},
                {"role": "user", "content": json.dumps(value, ensure_ascii=False)},
            ]
            if judge
            else value
        )
        kwargs = {
            "model": config["model"],
            "input": messages,
            "reasoning": {"effort": config["reasoning_effort"]},
            "max_output_tokens": config["max_output_tokens"],
            "store": False,
        }
        if judge:
            criteria = {
                name: {"type": "string", "enum": ["pass", "fail", "uncertain"]}
                for name in [
                    "meaning",
                    "terms_and_literals",
                    "mode",
                    "readability",
                    "output_discipline",
                ]
            }
            props = {
                "case_id": {"type": "string"},
                "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
                "critique": {"type": "string"},
                "critical_errors": {"type": "array", "items": {"type": "string"}},
                "criteria": {
                    "type": "object",
                    "properties": criteria,
                    "required": list(criteria),
                    "additionalProperties": False,
                },
            }
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "rewrite_verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": props,
                        "required": list(props),
                        "additionalProperties": False,
                    },
                }
            }
        response = client.responses.create(**kwargs)
        returned = response.model
        if returned != config["model"] and not returned.startswith(config["model"] + "-"):
            raise ValueError("Provider returned a different model")
        raw = response.model_dump(mode="json")
        reason = (
            "stop"
            if response.status == "completed"
            else (getattr(response.incomplete_details, "reason", None) or response.status)
        )
        result = {
            "text": response.output_text,
            "finish_reason": reason,
            "model": returned,
            "usage": raw.get("usage"),
            "provider_response_id": response.id,
            "raw_response": raw,
            "arka_revision": ARKA_REVISION,
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        }
        if judge:
            # Preserve a paid malformed response as data; the assessment fails closed later.
            try:
                judgment = json.loads(response.output_text)
                validate_judgment(value["case_id"], judgment)
                if reason != "stop":
                    raise ValueError("Judge output is incomplete")
                result["judgment"] = judgment
            except (ValueError, TypeError, KeyError):
                result["judgment_error"] = True
        return result

    return call


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[5]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "not-in-git-checkout", None
    return {
        "code_commit": os.environ.get("RUN_CODE_COMMIT", commit),
        "code_dirty": dirty,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/rewriter.toml"))
    parser.add_argument("--api-key-file", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", choices=["liquid", "luna"], required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--workers", type=int, default=1)
    judge = commands.add_parser("judge")
    judge.add_argument("--generation", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.api_key_file:
        os.environ["OPENROUTER_API_KEY"] = args.api_key_file.read_text().strip()
    settings = tomllib.loads(args.config.read_text())
    bundle = load_bundle(args.bundle)
    if any(c.source_kind == "personal_transcription" for c in bundle.cases):
        raise ValueError("This shared evaluation runner does not send personal transcriptions")
    if args.command == "generate":
        config = settings[args.model] | code_provenance()
        backend = liquid_backend(config) if args.model == "liquid" else hosted_backend(config)
        generate_cases(bundle, args.output, config, backend, workers=args.workers)
    else:
        root = args.bundle or Path(__file__).resolve().parents[3] / "data/rewriter/benchmark-draft"
        config = settings["judge"] | {"rubric": (root / "supporting/judge-system.txt").read_text()}
        judge_cases(
            bundle,
            args.generation,
            args.output,
            config,
            hosted_backend(config, judge=True),
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
