"""Scale native Arka pairs with independent Opus verification and frozen holdouts."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .arka_pipeline import (
    GENERATION_PROMPT,
    candidate_issues,
    parse_candidate,
    rescore_candidates,
    run_generation,
    write_verification_pack,
)
from .benchmark import MODES, Case, Rules
from .evaluation import digest, saved_call, write_json
from .training import Example, LeakageIndex, _tokenizer, token_audit, validation_index

PROJECT = Path(__file__).resolve().parents[3]
V2 = PROJECT / "data/rewriter/reviewed-v2"
VERIFIER = "anthropic/claude-opus-5.5"
TRAIN_DOMAINS = """a houseplant clinic|a woodworking makerspace|a community orchestra|
an urban birding group|a bicycle repair workshop|a language exchange|home baking|
a tabletop gaming club|a dog foster network|an archaeology museum|a school astronomy club|
a coastal cleanup|a neighborhood book exchange|urban allotments|a laundry cooperative|
a photo darkroom|a running club|train commuters|textile restoration|an independent bookstore|
a shared art studio|a local history newsletter|choir rehearsals|family travel planning|
board sports equipment|public library talks|amateur electronics|wheelchair sports|
community film nights|software localization|web accessibility|warehouse barcode scanning|
indoor climbing|garden pond care|a radio drama|a dance workshop|canoe hire|coffee roasting|
antique clocks|learning sign language|a student debating society|a small music fundraiser|
a veterinary reception desk|woodland mapping|collecting vinyl|adult swimming|
wildlife camera traps|flower arranging|furniture restoration|instrument lessons|
home energy monitoring|grocery deliveries|costume sewing|school lunches|personal budget records"""
DEV_DOMAINS = """dental appointments|an insect survey|kite building|a spice market stall|
a volcanic rock trail|parent reading sessions|a camping water filter|maritime flag history|
linocut printing|bus station wayfinding"""
TASKS = {
    "/slack": [
        "ask a specific question with useful context",
        "give an ordinary factual update",
        "coordinate routine work with colleagues",
        "describe a problem and ask for help",
        "politely decline a request and suggest an alternative",
        "reply to someone's feedback",
    ],
    "/transcribe": [
        "dictate a personal voice memo",
        "narrate a practical walkthrough",
        "dictate a message to another person",
        "record reminders in spoken order",
        "recount an experience",
        "think aloud about an everyday choice",
    ],
    "/prompt": [
        "request a practical plan",
        "request a comparison using supplied criteria",
        "request a draft using self-contained notes and quotations",
        "request instructions for a concrete task",
        "request a summary of supplied material",
        "request useful questions for a discussion",
    ],
    "/polish": [
        "write a personal reflection",
        "write a first-person review",
        "recount a meeting or visit",
        "explain a practical process",
        "write a friendly request",
        "write a factual announcement",
    ],
}


def expanded_plans():
    plans = []
    for mode in MODES:
        for split, text, count in [("train", TRAIN_DOMAINS, 6), ("dev", DEV_DOMAINS, 3)]:
            domains = [s.strip() for s in text.replace("\n", "").split("|")]
            for i in range(len(domains) * count):
                domain = domains[i // count]
                length = "short" if i % 10 < 5 else "medium" if i % 10 < 8 else "long"
                identity = i % 20 in {1, 6, 9}
                language = "de" if i % 20 == 3 else "pt" if i % 20 == 13 else "en"
                context = {}
                if mode == "/polish" and i % 5 == 0:
                    context["profile"] = [
                        "Plain and direct.",
                        "Friendly and conversational.",
                        "Clear and professional.",
                    ][i // 5 % 3]
                if mode == "/transcribe" and i % 6 == 5:
                    context["dictionary"] = [
                        ["NacreNote", "notes.yaml"],
                        ["VireoPad", "archive.csv"],
                        ["KestrelKit", "config.toml"],
                        ["MorrowDesk", "prefs.json"],
                    ][i // 6 % 4]
                case_id = f"arka-v3-{split}-{mode[1:]}-{i + 1:04}"
                brief = (
                    f"An independent fictional episode about {domain}: {TASKS[mode][i % count]}. "
                )
                if length == "long":
                    brief += (
                        "Use many substantive details, ordinary quotations and a numbered section. "
                    )
                    if mode == "/slack":
                        brief += "This is a pasted thread with distinct short posts and speakers. "
                if i % 5 == 0:
                    brief += "This is mostly definite ordinary information, not a dispute or unresolved investigation. "
                elif i % 5 == 1:
                    brief += "Keep one clear question or personal opinion when natural. "
                if identity:
                    brief += "Already correct input and exactly identical output. "
                brief += f"Episode {case_id}; do not put this identifier or authoring instructions in the text."
                plans.append(
                    {
                        "id": case_id,
                        "source_group": case_id,
                        "split": split,
                        "mode": mode,
                        "language": language,
                        "length": length,
                        "identity": identity,
                        "context": context,
                        "domain": domain,
                        "brief": brief,
                    }
                )
    return plans


def supplemental_plans():
    train_domains = """a pottery kiln booking|repairing a fountain pen|a skate maintenance class|
    a seed saving workshop|a neighborhood chess ladder|a ham radio exam|a puppet theatre|
    a rowing lesson|an apartment move|a carpool calendar|an audio walking tour|a bread starter|
    a fossil collection|a tennis lesson|a sewing machine repair|a sailing log|a pool maintenance log|
    a kitchen inventory|a street mural project|a camping meal plan|a volunteer helpline rota|
    a postcard exchange|a bicycle route map|a jigsaw puzzle library|a ceramic glaze test|
    a family recipe archive|a community print shop|a language pronunciation lesson|
    a motorbike tool kit|a stamp collection"""
    dev_domains = """an ice skating booking|a model railway exhibition|a firewood delivery|
    an olive tasting|a stargazing tent|a ferry ticket change|a sheepdog trial|
    a microscope loan|a weaving loom setup|an outdoor chess table"""
    domains = {
        s: [x.strip() for x in text.replace("\n", "").split("|")]
        for s, text in [("train", train_domains), ("dev", dev_domains)]
    }
    original = expanded_plans()
    plans = []
    for mode in MODES:
        for split, count, per_domain in [
            ("train", 120 if mode == "/transcribe" else 0, 4),
            ("dev", 20, 2),
        ]:
            selected = [p for p in original if p["split"] == split and p["mode"] == mode][:count]
            for i, source in enumerate(selected):
                case_id = f"arka-v3b-{split}-{mode[1:]}-{i + 1:04}"
                domain = domains[split][i // per_domain]
                brief = (
                    source["brief"].replace(source["domain"], domain).replace(source["id"], case_id)
                )
                plans.append(
                    {
                        **source,
                        "id": case_id,
                        "source_group": case_id,
                        "domain": domain,
                        "brief": brief,
                    }
                )
    return plans


def parse_reviews(packet, response):
    reviews = response.get("reviews", [])
    expected = {(x["id"], x["row_sha256"]) for x in packet}
    if len(reviews) != len(expected) or {(r["id"], r["row_sha256"]) for r in reviews} != expected:
        raise ValueError("review coverage or exact row identity differs")
    for r in reviews:
        values = [r[k] for k in ["input_verdict", "output_verdict", "pair_verdict"]]
        if any(x not in {"pass", "fail", "uncertain"} for x in values):
            raise ValueError("invalid review dimension")
        verdict = "fail" if "fail" in values else "uncertain" if "uncertain" in values else "pass"
        if verdict != r["verdict"] or not r["critique"].strip():
            raise ValueError("review dimensions contradict overall verdict or lack critique")
    return reviews


def select_balanced(pool, *, train_per_mode=250, dev_per_mode=25):
    selected = [p for p in pool if p[1]["split"] == "validation"]
    seen = {r["output"].strip().casefold() for r, m in selected}
    for split, count in [("train", train_per_mode), ("dev", dev_per_mode)]:
        for mode in MODES:
            choices = [p for p in pool if p[1]["split"] == split and p[0]["mode"] == mode]
            choices.sort(key=lambda p: (not p[1].get("carried_from_v2", False), p[1]["id"]))
            retained = []
            for row, meta in choices:
                target = row["output"].strip().casefold()
                if target not in seen:
                    retained.append((row, meta))
                    seen.add(target)
                if len(retained) == count:
                    break
            if len(retained) < count:
                raise ValueError(f"insufficient {split} {mode}: {len(retained)} of {count}")
            selected.extend(retained)
    if len({m["id"] for r, m in selected}) != len(selected):
        raise ValueError("duplicate release identity")
    return selected


VERIFY_SYSTEM = """Verify rewrite DATA ONLY. Do not write improved examples or solve the user's task.
Read every input and target in full. Return one review per exact id and row_sha256.
input_verdict checks plausible natural drafts or dictation, recoverable intent, contradictions,
fake noise, padding and leaked authoring instructions. output_verdict checks natural appropriate
language and formatting. pair_verdict checks ALL facts, uncertainty, conditions, negation,
quantities, names, quotations, literals, requests versus assertions, and the supplied mode.
/transcribe permits punctuation, true disfluency removal and explicit corrections, not lexical
paraphrasing. Preserve meaningful words like German 'um' and 'er', colloquial Portuguese and
unfinished thoughts. /slack is casual minimal cleanup; a long input may be a pasted thread.
/prompt reorganizes a self-contained request and never answers it. It must not drop supplied
material, hedges or constraints. /polish improves prose without omissions or changed certainty.
Identity inputs must remain byte-identical. Apply only supplied dictionary/profile context.
Check long texts paragraph by paragraph. Fail concrete defects. Uncertain means not confidently
justifiable. Pass requires no material defect. Overall verdict is fail if any dimension fails,
uncertain if any dimension is uncertain and none fails, otherwise pass. Give a concrete critique
for EVERY pair, citing the changed detail for defects. These are model judgments, not human labels.
"""


def review_candidates(
    root: Path,
    workers=8,
    *,
    retry_failed=False,
    model=VERIFIER,
    prefix="opus",
    batch_size=8,
    max_tokens=6000,
    system_prompt=VERIFY_SYSTEM,
    temperature: float | None = None,
):
    from arka.config.models import LLMConfig
    from arka.llm.openai_client import build_openai_client

    source = root / "generation"
    rows = [json.loads(x) for x in (source / "candidates.jsonl").read_text().splitlines()]
    scored = source / "scored-supporting.json"
    meta = json.loads((scored if scored.exists() else source / "supporting.json").read_text())
    old_rows = [
        json.loads(line)
        for split in ["train", "dev", "validation"]
        for line in (V2 / f"data/{split}.jsonl").read_text().splitlines()
    ]
    frozen = LeakageIndex([r[k] for r in old_rows for k in ["input", "output"]])
    protected = validation_index()
    clean = []
    for row, m in zip(rows, meta, strict=True):
        if len(m["decisions"]) != 6:
            m["mechanical_issues"].append("incomplete_native_decisions")
        if any(frozen.matches(row[k]) or protected.matches(row[k]) for k in ["input", "output"]):
            m["mechanical_issues"].append("v2_or_original_or_personal_overlap")
        if not m["mechanical_issues"]:
            clean.append((row, m))
    write_json(root / "candidate-supporting.json", meta)
    write_verification_pack([r for r, m in clean], [m for r, m in clean], root / "blind-packets")
    packet = [
        {
            "id": m["id"],
            "row_sha256": digest(r),
            "mode": r["mode"],
            "language": m["language"],
            "context": m["context"],
            "identity": m["identity"],
            "mode_instructions": MODES[r["mode"]],
            "input": r["input"],
            "output": r["output"],
        }
        for r, m in clean
    ]
    properties = {k: {"type": "string"} for k in ["id", "row_sha256", "critique"]}
    properties.update(
        {
            k: {"type": "string", "enum": ["pass", "fail", "uncertain"]}
            for k in ["input_verdict", "output_verdict", "pair_verdict", "verdict"]
        }
    )
    schema = {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }
    if prefix not in {"opus", "sol"} or batch_size < 1:
        raise ValueError("Invalid verifier prefix or batch size")
    client = build_openai_client(
        LLMConfig(
            provider="openai",
            model=model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout_seconds=240,
            max_retries=1,
        )
    )
    batches = [packet[i : i + batch_size] for i in range(0, len(packet), batch_size)]

    calls_dir = root / f"{prefix}-calls"
    calls_by_id = {}
    if calls_dir.exists():
        for cp in calls_dir.glob("*.json"):
            try:
                crec = json.loads(cp.read_text(encoding="utf-8"))
                if crec.get("status") == "completed":
                    content = crec["result"]["choices"][0]["message"]["content"]
                    rev = json.loads(content)
                    if rev.get("reviews") and len(rev["reviews"]) == 1:
                        cid = rev["reviews"][0]["id"]
                        calls_by_id[cid] = (cp, crec.get("request_sha256"))
            except (json.JSONDecodeError, OSError, KeyError, IndexError, TypeError):
                pass

    def read_call(path, request, batch):
        result = saved_call(
            path,
            request,
            lambda: client.chat.completions.create(**request).model_dump(mode="json"),
        )
        try:
            if result["status"] != "completed":
                return []
            returned = result["result"].get("model", "")
            if returned != model and not returned.startswith(model + "-"):
                return []
            choice = result["result"]["choices"][0]
            if choice["finish_reason"] != "stop":
                return []
            return parse_reviews(batch, json.loads(choice["message"]["content"]))
        except (ValueError, KeyError, TypeError):
            return []

    def one(item):
        index, batch = item
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "rewrite_reviews", "strict": True, "schema": schema},
            },
        }
        if temperature is not None:
            request["temperature"] = temperature
        if len(batch) == 1:
            request = copy.deepcopy(request)
            array = request["response_format"]["json_schema"]["schema"]["properties"]["reviews"]
            array["minItems"] = array["maxItems"] = 1
            for key in ["id", "row_sha256"]:
                array["items"]["properties"][key]["enum"] = [batch[0][key]]
        req_hash = digest(request)
        cid = batch[0]["id"] if len(batch) == 1 else None
        if cid and cid in calls_by_id and calls_by_id[cid][1] == req_hash:
            paths = [calls_by_id[cid][0]]
        else:
            cand_name = cid if cid else f"{index:04}"
            cand_path = root / f"{prefix}-calls/{cand_name}.json"
            if cand_path.exists():
                try:
                    existing_rec = json.loads(cand_path.read_text(encoding="utf-8"))
                    if existing_rec.get("request_sha256") != req_hash:
                        cand_path.unlink()
                except (json.JSONDecodeError, OSError, KeyError):
                    cand_path.unlink(missing_ok=True)
            paths = [cand_path]
            paths += sorted((root / f"{prefix}-calls").glob(f"{cand_name}-retry-*.json"))
        reviews = []
        for path in paths:
            reviews = read_call(path, request, batch)
            if reviews:
                break  # Never re-judge a completed verdict, including fail or uncertain.
        if not reviews and retry_failed:
            # Retry a malformed or interrupted batch as independent one-row
            # packets. Constrain identity copying without changing the rubric.
            singles = []
            for position, entry in enumerate(batch):
                single_request = copy.deepcopy(request)
                single_request["messages"][1]["content"] = json.dumps([entry], ensure_ascii=False)
                array = single_request["response_format"]["json_schema"]["schema"]["properties"][
                    "reviews"
                ]
                array["minItems"] = array["maxItems"] = 1
                for key in ["id", "row_sha256"]:
                    array["items"]["properties"][key]["enum"] = [entry[key]]
                base = f"{index:04}-single-{position:02}"
                saved = sorted((root / f"{prefix}-calls").glob(base + "-*.json"))
                single = []
                for path in saved:
                    single = read_call(path, single_request, [entry])
                    if single:
                        break
                if not single:
                    single = read_call(
                        root / f"{prefix}-calls/{base}-{len(saved):04}.json",
                        single_request,
                        [entry],
                    )
                singles.extend(single)
            if len(singles) == len(batch):
                reviews = singles
        if not reviews:
            print(f"{prefix} batch {index}: incomplete; explicit retry required", flush=True)
            return [], index
        print(
            f"{prefix} batch {index + 1}/{len(batches)}: {dict(Counter(r['verdict'] for r in reviews))}",
            flush=True,
        )
        return reviews, None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(one, enumerate(batches)))
    reviews = [r for batch, _ in results for r in batch]
    failed = [index for _, index in results if index is not None]
    report = {
        "reviewer_model": model,
        "scope": "blind exact-row input, output and pair verification",
        "coverage_count": len(reviews),
        "reviews": reviews,
        "human_reviewed": False,
        "incomplete_batch_indices": failed,
    }
    write_json(root / f"{prefix}-reviews.json", report)
    return {
        "candidates": len(rows),
        "mechanically_clean": len(clean),
        "reviewed": len(reviews),
        "verdicts": dict(Counter(r["verdict"] for r in reviews)),
        "incomplete_batch_indices": failed,
    }


def decode_generated(raw, plan, expected_seed):
    if raw["seed_id"] != expected_seed:
        raise ValueError("generated response belongs to a different seed")
    value = json.loads(raw["generated_text"])
    return parse_candidate(plan, value["instruction"], value["response"])


def review_ready(root: Path, limit=128, *, resume=False):
    """Freeze completed source calls into one immutable shard; never read partial results."""
    from arka.pipeline.output import OutputWriter

    source = root / "generation"
    plans = json.loads((source / "source-plan.json").read_text())["plans"]
    seeds = OutputWriter().read_parquet(source / "runs/author/stages/01_source/data.parquet")
    consumed = set()
    shard_base = root / "review-shards"
    pending = None
    for shard in sorted(shard_base.glob("*")):
        summary_path = shard / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        incomplete = not summary or summary.get("reviewed", 0) < summary.get(
            "mechanically_clean", 0
        )
        if incomplete:
            if not resume:
                raise ValueError(f"Resume incomplete shard explicitly with --resume: {shard}")
            pending = shard
            break
    for path in shard_base.glob("*/source-indices.json"):
        consumed.update(json.loads(path.read_text()))
    raw_path = source / "runs/author/stages/02_generate/raw_responses.jsonl"
    text = raw_path.read_text()
    lines = text.splitlines()
    if not text.endswith("\n"):
        lines = lines[:-1]
    if pending and (pending / "completed-source-calls.json").exists():
        ready = json.loads((pending / "completed-source-calls.json").read_text())
        raw_by_index = {json.loads(line)["plan_index"]: json.loads(line) for line in lines}
        if any(raw_by_index.get(r["plan_index"]) != r for r in ready):
            raise ValueError("Incomplete shard sources no longer match the raw generation log")
    else:
        ready = []
        for line in lines:
            raw = json.loads(line)
            if raw["plan_index"] not in consumed:
                ready.append(raw)
            if len(ready) == limit:
                break
    if not ready:
        return {"new_completed_sources": 0, "previously_claimed": len(consumed)}
    shard = pending or shard_base / f"{len(list(shard_base.glob('*/source-indices.json'))):04}"
    shard.mkdir(parents=True, exist_ok=bool(pending))
    # Save the complete source snapshot before claiming its indices.
    write_json(shard / "completed-source-calls.json", ready)
    write_json(shard / "source-indices.json", [r["plan_index"] for r in ready])
    rows, metadata, rejected = [], [], []
    tokenizer = _tokenizer()
    for raw in ready:
        index = raw["plan_index"]
        plan = plans[index]
        try:
            row = decode_generated(raw, plan, seeds[index].id)
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
            rows.append(row)
            metadata.append(
                {
                    **plan,
                    "row_sha256": digest(row),
                    "record_id": raw["seed_id"],
                    "raw_response_sha256": digest(raw),
                    "mechanical_issues": candidate_issues(plan, row),
                    "decisions": {},
                    "tokens": {k: v for k, v in tokens.items() if k not in {"input_ids", "labels"}},
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"id": plan["id"], "error_type": type(error).__name__})
    data = shard / "generation"
    data.mkdir(exist_ok=True)
    (data / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    write_json(data / "supporting.json", metadata)
    write_json(data / "rejected.json", rejected)
    if rows:
        if not (data / "scored-supporting.json").exists():
            rescore_candidates(data, data / "native-checks")
        summary = review_candidates(shard, retry_failed=bool(pending))
    else:
        summary = {"candidates": 0, "reviewed": 0}
    summary.update(
        shard=str(shard), completed_sources=len(ready), token_or_schema_rejected=len(rejected)
    )
    write_json(shard / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "review", "review-ready"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--plans-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.stage == "generate":
        prompt = GENERATION_PROMPT.replace(
            "1000 to 1300 substantive words", "800 to 950 substantive words"
        )
        plans = supplemental_plans() if args.supplement else expanded_plans()
        if args.plans_file:
            if args.supplement or args.pilot:
                raise ValueError("Use a plans file separately from built-in plan variants")
            plans = json.loads(args.plans_file.read_text())
        if args.supplement:
            prompt = prompt.replace(
                "Scenario:\n",
                """For /transcribe, compare the words in the target with the raw dictation before returning.
Do not add articles or pronouns, remove conjunctions, expand colloquial words, or turn indirect
speech into direct speech. Retain the original lexical choices. A correction may discard the
explicitly superseded phrase, but do not leave the correction cue or change surrounding words.
Not every example needs a correction; ordinary missing punctuation is useful too. Identity
examples must already have correct capitalization and formatting, with no unresolved corrections.

Scenario:
""",
            )
        if args.pilot:
            plans = [p for mode in MODES for p in [p for p in plans if p["mode"] == mode][:16]]
        result = run_generation(
            args.run / "generation",
            plans_override=plans,
            prompt_override=prompt,
            resume=args.resume,
            workers=8,
        )
    elif args.stage == "review-ready":
        result = review_ready(args.run, args.batch_size, resume=args.resume)
    else:
        result = review_candidates(args.run)
    write_json(args.run / f"{args.stage}-summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
