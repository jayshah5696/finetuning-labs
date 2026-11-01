"""Target observed rewrite failure types with new Arka pairs and exact Sol verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from .arka_pipeline import DIMENSIONS, GENERATION_PROMPT, candidate_issues, rubric, run_generation
from .benchmark import MODES
from .evaluation import digest, write_json
from .scale_data import VERIFY_SYSTEM, parse_reviews, review_candidates
from .training import LeakageIndex, validation_index

PROJECT = Path(__file__).resolve().parents[3]
CURRENT = PROJECT / "data/rewriter/current"
SOL = "openai/gpt-6.1-sol"
QUOTAS = {
    "/slack": {"short": 15, "medium": 15, "long": 20},
    "/transcribe": {"short": 100, "medium": 50, "long": 50},
    "/prompt": {"short": 30, "medium": 60, "long": 60},
    "/polish": {"short": 20, "medium": 30, "long": 50},
}
DOMAINS = """a repair cafe|a greenhouse watering rota|a small theatre lighting crew|a shared bicycle shed|
a college film society|a household parcel pickup|a neighborhood weather station|a ceramics delivery|
an apartment intercom repair|a screen-reader audit|a seed exchange|a local swimming timetable|
a music lesson booking|a ferry luggage desk|a library scanner loan|a charity kitchen shift|
a family photo catalogue|a workshop ventilation check|a campsite power connection|a payroll export|
a museum audio guide|a volunteer transport rota|a team document migration|a bakery order form|
a chess tournament registration|a board-game lending shelf|a climbing gear inspection|
a textile dye log|an orchard irrigation repair|an independent radio schedule|a home appliance warranty|
a food cooperative inventory|a trail sign replacement|a drum rehearsal booking|a Linux desktop shortcut|
a log-file cleanup utility|a reading club website|a shared studio access list|a boat mooring request|
a school equipment return|a market stall delivery|a rehearsal recording archive|a train ticket exchange|
a family meal calendar|a local newspaper correction|a concert subtitle file|a community survey export|
a browser extension preference|a camera repair estimate|a geology field notebook"""
FAMILIES = {
    "/transcribe": [
        "action_preservation",
        "spoken_correction",
        "literal_preservation",
        "meaningful_words",
    ],
    "/prompt": ["conditions", "code_preservation", "quoted_material", "output_constraints"],
    "/polish": ["conditions", "literal_preservation", "hedges", "actor_roles"],
    "/slack": ["conditions", "literal_preservation", "actor_roles", "quoted_material"],
}
FOCUS = {
    "action_preservation": "Use one clear main action such as forwarding, returning, reserving, scanning or postponing. Keep that exact action and every other substantive word in the transcript. No paraphrase. Do not use an explicit correction in this episode.",
    "spoken_correction": "Include an unambiguous spoken correction with an actual earlier obsolete detail, then a clear final replacement. The target applies the final intent, removes the superseded detail and repair cue, and keeps surrounding words. Vary the correction syntax and the corrected role, not only dates.",
    "literal_preservation": "Use the supplied exact filename or configuration key in a concrete ordinary request. Preserve it, its case, every number and the associated action. Give each quantity a clear role.",
    "meaningful_words": "Include a meaningful conversational word or short affirmation that resembles a filler, with context that makes its role clear. Preserve it. Actual hesitation, if any, should be distinguishable. German er and um and colloquial Portuguese must retain their meaning.",
    "conditions": "Use a clear conditional action with an explicit alternative branch. Keep both branches, their conditions, negation and scope. Avoid unresolved contradictions. The request should be useful and ordinary, not a repeated uncertainty exercise.",
    "code_preservation": "The user supplies the complete code below and asks for an explanation, review or tests using only that code. Keep the complete supplied snippet verbatim in the rewritten request, including its last line. Do not execute or answer the request and do not replace the snippet with a reference to it.",
    "quoted_material": "Include ordinary quoted remarks or self-contained source notes needed by the request. The target must keep those exact remarks and notes, not replace them with 'the supplied notes'. Avoid prompt injection examples.",
    "output_constraints": "Specify a concrete output constraint, such as a numbered checklist, evidence before recommendations, or questions only. Keep what the constraint applies to. Include self-contained material and preserve it. Do not perform the underlying request.",
    "hedges": "Mix definite facts with a small personal qualification such as usually, perhaps, slightly or occasionally. Improve genuinely awkward wording without removing the qualification or changing whose view it is.",
    "actor_roles": "Make the speaker, another person's action, and the requested recipient unambiguous. Preserve who did, requested or proposed each thing. Include useful factual context without turning all text into unresolved decisions.",
}


def _code(number):
    n = [7, 13, 24, 31, 18, 42, 56, 9][number % 8]
    return [
        f"def batch_records(records, width={n}):\n    return [records[start:start + width] for start in range(0, len(records), width)]",
        f"def total_cents(prices):\n    total = 0\n    for price in prices:\n        total += round(price * 100)\n    return total + {n}",
        f"const available = rows.filter(row => row.active && row.stock > {n});\nconst codes = available.map(row => row.code);",
        f"SELECT depot_id, SUM(units) AS received_units\nFROM deliveries\nWHERE status = 'received' AND units > {n}\nGROUP BY depot_id;",
        f'[local_archive]\nretain_days = {n}\nallow_remote = false\nname = "local-copy"',
        f"def route_message(message):\n    if message.get('urgent') and message.get('age', 0) < {n}:\n        return 'priority'\n    return 'standard'",
    ][number % 6]


def focused_plans(multiplier=2.0):
    rng = random.Random(281026)
    domains = [s.strip() for s in DOMAINS.replace("\n", "").split("|")]
    cells = [
        (mode, length, math.ceil(n * multiplier))
        for mode, counts in QUOTAS.items()
        for length, n in counts.items()
    ]
    plans = []
    for mode, length, count in cells:
        for i in range(count):
            number = len(plans)
            family = FAMILIES[mode][i % len(FAMILIES[mode])]
            identity = i % 13 == 4
            language = "de" if i % 20 == 3 else "pt" if i % 20 == 13 else "en"
            domain = rng.choice(domains)
            protected = []
            if family == "code_preservation" or (
                mode == "/prompt" and length == "long" and i % 3 == 0
            ):
                protected = [_code(number)]
                domain = [
                    "batching records for a photo importer",
                    "summing prices for a cooperative shop",
                    "filtering stock in a small web shop",
                    "summarizing deliveries for a depot",
                    "configuring a family photo archive",
                    "routing messages for a volunteer help desk",
                ][number % 6]
            elif family == "literal_preservation":
                protected = [
                    [
                        "speaker-rota.csv",
                        "north-gate-notes.md",
                        "delivery-summary.tsv",
                        "lighting-cues.yaml",
                        "winter-kit-list.txt",
                    ][i % 5]
                    if i % 2
                    else [
                        "transfer.allow_resume",
                        "preview.show_labels",
                        "archive.retain_originals",
                        "sync.limit_uploads",
                        "reader.prefer_local",
                    ][i % 5]
                ]
                if i % 2 == 0:
                    domain = [
                        "configuring a desktop uploader",
                        "setting up an archive preview",
                        "maintaining a family photo importer",
                        "managing a shared-folder sync client",
                        "configuring an offline reading app",
                    ][i % 5]
            context = {}
            if mode == "/polish":
                context["profile"] = [
                    "Plain and direct.",
                    "Friendly and conversational.",
                    "Clear and professional.",
                ][i % 3]
            if mode == "/transcribe" and i % 9 == 2:
                names = [
                    "MallowNote",
                    "BrackenPad",
                    "KiteLedger",
                    "PebbleDesk",
                    "LumenCrate",
                    "HarborClip",
                    "CedarCue",
                    "BirchIndex",
                    "RowanSync",
                    "WrenQueue",
                    "SpruceGate",
                    "TideBoard",
                    "LarchVault",
                    "IrisLedger",
                    "AsterBrief",
                ]
                context["dictionary"] = [names[i % len(names)]]
            brief = f"Create an independent fictional episode about {domain}. {FOCUS[family]} "
            if identity:
                brief = (
                    "Create an already correct, self-contained text about "
                    + domain
                    + ". The output is exactly identical to the input. No repairs are needed. "
                )
            if length == "long":
                brief += "Use many substantive, varied facts across connected paragraphs and a numbered section. Keep a substantive final paragraph. Include ordinary direct quotations and at least one condition that matters later. Preserve the complete final section in the output. "
                if mode == "/slack":
                    brief += "Write a pasted thread of short posts with different speaker labels. "
            if protected:
                brief += (
                    "Include this exact supplied material in BOTH raw input and target, with all line breaks intact:\n"
                    + "\n\n".join(protected)
                )
            case_id = f"arka-boost-{mode[1:]}-{length}-{i + 1:04}"
            plans.append(
                {
                    "id": case_id,
                    "source_group": case_id,
                    "split": "train",
                    "mode": mode,
                    "language": language,
                    "length": length,
                    "identity": identity,
                    "context": context,
                    "domain": domain,
                    "task_family": "long_completeness" if length == "long" else family,
                    "focus": family,
                    "protected_literals": protected,
                    "brief": brief,
                }
            )
    # Interleave cells so every batch covers more than one mode and length tier.
    rng.shuffle(plans)
    return plans


def strong_decisions(meta):
    decisions = meta.get("decisions", {})
    return bool(
        meta.get("native_accepted")
        and set(decisions) == {"labeling_engine." + k for k in DIMENSIONS}
        and all(
            v.get("score", 0) >= 0.8 and v.get("confidence", 0) >= 0.2 for v in decisions.values()
        )
    )


def complete_decisions(meta):
    decisions = meta.get("decisions", {})
    return set(decisions) == {"labeling_engine." + k for k in DIMENSIONS} and all(
        isinstance(v.get(k), (int, float)) and math.isfinite(v[k]) and 0 <= v[k] <= 1
        for v in decisions.values()
        for k in ["score", "confidence"]
    )


def native_rejection(meta):
    return any(
        v.get("score", 1) <= 0.1 and v.get("confidence", 0) >= 0.8
        for v in meta.get("decisions", {}).values()
    )


def verified_pair(row, meta, review, *, second_review=None):
    def exact_pass(value):
        return bool(
            value
            and value.get("id") == meta["id"]
            and value.get("row_sha256") == digest(row) == meta["row_sha256"]
            and all(
                value.get(k) == "pass"
                for k in ["input_verdict", "output_verdict", "pair_verdict", "verdict"]
            )
        )

    return bool(
        not meta.get("mechanical_issues")
        and complete_decisions(meta)
        and not native_rejection(meta)
        and exact_pass(review)
        and (strong_decisions(meta) or exact_pass(second_review))
    )


def boost_issues(plan, row, blocked):
    if "family" in plan or plan.get("source_group", "").startswith("expansion-"):
        from .diverse_expansion import validate_diverse_candidate

        issues = validate_diverse_candidate(row["input"], row["output"], plan.get("context"), plan)
        if blocked.matches(row["input"]) or blocked.matches(row["output"]):
            issues.append("protected_overlap")
        if plan["id"] in row["input"] + row["output"]:
            issues.append("source_id_leak")
        return issues

    from .english_critical import critical_issues

    issues = candidate_issues(plan, row)
    issues.extend(critical_issues(plan, row))
    if blocked.matches(row["input"]) or blocked.matches(row["output"]):
        issues.append("protected_overlap")
    if any(
        literal not in row["input"] or literal not in row["output"]
        for literal in plan.get("protected_literals", [])
    ):
        issues.append("protected_literal_changed")
    if plan["id"] in row["input"] + row["output"]:
        issues.append("source_id_leak")
    if (
        row["mode"] == "/transcribe"
        and not plan.get("identity")
        and plan.get("task_family") == "action_preservation"
    ):

        def words(text):
            fillers = {
                "en": {"uh", "um", "erm", "er", "ah"},
                "de": {"äh", "ähm"},
                "pt": {"ahn", "hum"},
            }[plan["language"]]
            tokens = [w for w in re.findall(r"\w+", text.casefold()) if w not in fillers]
            return [w for i, w in enumerate(tokens) if i == 0 or w != tokens[i - 1]]

        if words(row["input"]) != words(row["output"]):
            issues.append("transcribe_lexical_change")
    return issues


ADJUDICATION_SYSTEM = (
    "You are an independent second-round expert reviewer evaluating rewritten conversational requests.\n"
    "Provide an independent, fresh critical analysis evaluating the candidate rewrite against the criteria.\n"
    "Strictly assess input fidelity, output quality, pair validity, and mode requirements.\n"
    "Give a concrete critique for EVERY pair, citing specific details."
)


def protected_index():
    rows = [
        json.loads(line)
        for split in ["train", "dev", "validation"]
        for line in (CURRENT / f"data/{split}.jsonl").read_text().splitlines()
    ]
    index = validation_index()
    return LeakageIndex([*index.texts, *[r[k] for r in rows for k in ["input", "output"]]])


def review_packet(row, meta):
    return {
        "id": meta["id"],
        "row_sha256": digest(row),
        "mode": row["mode"],
        "language": meta["language"],
        "context": meta["context"],
        "identity": meta["identity"],
        "mode_instructions": MODES[row["mode"]],
        "input": row["input"],
        "output": row["output"],
    }


def verify_holdouts(contents, freeze):
    if any(
        name not in contents or hashlib.sha256(contents[name]).hexdigest() != sha
        for name, sha in freeze.items()
    ):
        raise ValueError("Frozen dev or validation holdout bytes changed")


def verify_overlap_report(rows, report):
    if (
        report.get("rows") != len(rows)
        or report.get("new_to_new_overlap") != 0
        or report.get("rows_sha256") != digest(rows)
    ):
        raise ValueError("Final overlap check does not match the exact additions")


def verify_saved_review(row, meta, record):
    if (
        record["status"] != "completed"
        or digest(record["request"]) != record["request_sha256"]
        or digest(record["result"]) != record["result_sha256"]
    ):
        raise ValueError("Incomplete or changed verification request/result")
    request, result = record["request"], record["result"]
    if request["model"] != SOL or result.get("model") != SOL:
        raise ValueError("Unapproved verification model")
    if (
        request["messages"][0]["role"] != "system"
        or request["messages"][0]["content"] not in {VERIFY_SYSTEM, ADJUDICATION_SYSTEM}
        or json.loads(request["messages"][1]["content"]) != [review_packet(row, meta)]
    ):
        raise ValueError("Verification packet differs from exact row")
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Incomplete verification output")
    return parse_reviews([review_packet(row, meta)], json.loads(choice["message"]["content"]))[0]


def focused_rubric():
    value = rubric()
    additions = {
        "candidate_input": "The MODE/LANGUAGE/PROFILE/DICTIONARY header is context, not raw prose. Imperfect spoken wording and dictated corrections are valid. A factual update need not contain a request. Judge naturalness in its stated genre, not whether it is already polished.",
        "output_quality": "An unchanged clean input is valid. Do not demand a new fact, an answer to a prompt, or expanded prose. A deliberate unfinished thought may remain unfinished.",
        "pair_facts": "Preserve the FINAL intended facts. Details explicitly superseded by a spoken correction MUST disappear. Their removal is not an omission. If input and response are identical, this dimension is satisfied. Judge the source after INPUT, not header labels.",
        "pair_intent": "Apply the relevant speech act only: statements may remain statements and opinions may remain opinions. An episode with no proposal or question is not required to invent one. For identical text this dimension is satisfied. For a correction, preserve final intent rather than every obsolete phrase.",
        "pair_literals": "Compare the exact quoted text, supplied code and final intended names/numbers. Remove explicitly superseded corrected details. Do not require a cancelled number to survive. Identical text satisfies preservation.",
        "pair_mode": "Use the explicit MODE only. A clean identity input may remain exactly unchanged. Minimal punctuation/capitalization repairs are useful, not failures. Prompt outputs are requests and must retain their supplied material; they must not fulfill the underlying task.",
    }
    value["version"] = "fidelity-targeted"
    for dimension in value["dimensions"]:
        dimension["description"] = additions[dimension["name"]] + " " + dimension["description"]
        if dimension["name"] == "pair_facts":
            dimension["description"] = (
                "Does the response preserve all final intended facts, sections, conditions, negation and certainty from the text after INPUT, with no invented facts or missing sections? Apply explicit corrections: a superseded detail must be removed, not preserved. Identical input/response preserves meaning."
            )
        if dimension["name"] == "pair_intent":
            dimension["description"] = (
                "Does the response preserve the speech act actually present in the INPUT: statement, question, request, opinion or proposal? Do not require a question or proposal where none exists. Apply final intended corrections. A /prompt rewrite is a request, never its answer. Identical text satisfies this dimension."
            )
    return value


def run_batch(root, plans):
    prompt = GENERATION_PROMPT.replace("20 to 100 words", "5 to 60 words").replace(
        "1000 to 1300 substantive words", "900 to 1000 substantive words"
    )
    prompt = prompt.replace(
        "Scenario:\n",
        """For the targeted fidelity examples, preserve every supplied code line and exact literal.
Do not weaken output constraints, remove an alternative condition, or swap a main action.
Keep the task and domain coherent. Do not pad a short request with many repetitive negative
constraints or invent new restrictions in the target. Vary the prose and episode structure.
Compare the final paragraph and final code line before returning the target. Do not append
an invented summary, approval or recommendation. Briefs describe new fictional episodes only.

Scenario:
""",
    )
    data = root / "generation"
    if not (data / "summary.json").exists():
        summary = run_generation(
            data,
            plans_override=plans,
            prompt_override=prompt,
            workers=16,
            rubric_override=focused_rubric(),
            resume=(data / "source-plan.json").exists(),
        )
        write_json(data / "summary.json", summary)
    rows = [json.loads(line) for line in (data / "candidates.jsonl").read_text().splitlines()]
    meta = json.loads((data / "supporting.json").read_text())
    blocked = protected_index()
    for row, m in zip(rows, meta, strict=True):
        m["mechanical_issues"] = sorted(set(m["mechanical_issues"] + boost_issues(m, row, blocked)))
        if not complete_decisions(m):
            m["mechanical_issues"].append("incomplete_native_decisions")
        if native_rejection(m):
            m["mechanical_issues"].append("confident_native_rejection")
    write_json(data / "scored-supporting.json", meta)
    summary = review_candidates(root, model=SOL, prefix="sol", batch_size=1, max_tokens=4096)
    if summary["incomplete_batch_indices"]:
        summary = review_candidates(
            root, model=SOL, prefix="sol", batch_size=1, max_tokens=4096, retry_failed=True
        )
    reviews = json.loads((root / "sol-reviews.json").read_text())["reviews"]
    mapping = {(v["id"], v["row_sha256"]): v for v in reviews}
    uncertain = []
    for row, m in zip(rows, meta, strict=True):
        first = mapping.get((m["id"], digest(row)))
        if (
            not m["mechanical_issues"]
            and first
            and first["verdict"] == "pass"
            and not strong_decisions(m)
        ):
            uncertain.append((row, m))
    second_mapping = {}
    if uncertain:
        second_root = root / "adjudication"
        second_data = second_root / "generation"
        second_data.mkdir(parents=True, exist_ok=True)
        (second_data / "candidates.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r, m in uncertain)
        )
        write_json(second_data / "scored-supporting.json", [m for r, m in uncertain])
        second_summary = review_candidates(
            second_root, model=SOL, prefix="sol", batch_size=1, max_tokens=4096
        )
        if second_summary["incomplete_batch_indices"]:
            review_candidates(
                second_root,
                model=SOL,
                prefix="sol",
                batch_size=1,
                max_tokens=4096,
                retry_failed=True,
            )
        second_reviews = json.loads((second_root / "sol-reviews.json").read_text())["reviews"]
        second_mapping = {(v["id"], v["row_sha256"]): v for v in second_reviews}
    selected = []
    for row, m in zip(rows, meta, strict=True):
        review = mapping.get((m["id"], digest(row)))
        second = second_mapping.get((m["id"], digest(row)))
        if verified_pair(row, m, review, second_review=second):
            selected.append((row, m | {"adjudication_review": second}, review))
    write_json(root / "summary.json", summary | {"verified_passes": len(selected)})
    return selected


def validate_selection(entries, *, quotas=QUOTAS):
    ids, targets, counts = set(), set(), Counter()
    for entry in entries:
        row, meta, review = entry["row"], entry["meta"], entry["review"]
        target = " ".join(row["output"].casefold().split())
        if meta["id"] in ids or target in targets:
            raise ValueError("duplicate addition")
        ids.add(meta["id"])
        targets.add(target)
        counts[row["mode"], meta.get("length")] += 1
        if not verified_pair(row, meta, review, second_review=meta.get("adjudication_review")):
            raise ValueError("Addition lacks exact passing verification")
    if counts != Counter(
        {(m, length): n for m, cells in quotas.items() for length, n in cells.items()}
    ):
        raise ValueError("Addition quotas are incomplete")


def supplement_plans(counts, attempt):
    remaining = Counter(
        {
            (m, length): max(0, n - counts.get((m, length), 0)) * 6
            for m, cells in QUOTAS.items()
            for length, n in cells.items()
        }
    )
    plans = []
    for plan in focused_plans(10):
        key = (plan["mode"], plan["length"])
        if remaining[key] <= 0:
            continue
        plan["id"] = plan["id"].replace("arka-boost-", f"arka-boost-extra{attempt}-")
        plan["source_group"] = plan["id"]
        plan["brief"] += (
            " Use fresh fictional people, details and wording for this independent episode."
        )
        if attempt >= 2 and plan["mode"] == "/prompt" and plan["length"] == "long":
            plan["brief"] += (
                " Put the substantive source information in a self-contained reference-notes block, "
                "separate from a concise rough request. The notes contain ordinary concrete events, "
                "quotations and any supplied code, not a long checklist of negative constraints. "
                "Improve and organize the request, and copy the entire reference-notes block "
                "verbatim into the target. Do not paraphrase source material or refer to missing notes."
            )
        plans.append(plan)
        remaining[key] -= 1
    return plans


def finish_boost(run):
    """Remove new-to-new overlaps and fill missing cells with fresh, fully reviewed episodes."""
    run = run.resolve()
    entries = json.loads((run / "retained.json").read_text())
    exclusions_path = run / "overlap-exclusions.json"
    retained, texts = [], []
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else []
    for entry in entries:
        row = entry["row"]
        index = LeakageIndex(texts)
        if index.matches(row["input"]) or index.matches(row["output"]):
            exclusions.append({"id": entry["meta"]["id"], "reason": "new_to_new_overlap"})
        else:
            retained.append(entry)
            texts.extend([row["input"], row["output"]])
    counts = Counter((e["row"]["mode"], e["meta"]["length"]) for e in retained)
    seen_ids = {e["meta"]["id"] for e in retained}
    for attempt in range(1, 7):
        if len(retained) == 500:
            break
        plan_path = run / f"extra-plans-{attempt}.json"
        plans = (
            json.loads(plan_path.read_text())
            if plan_path.exists()
            else supplement_plans(counts, attempt)
        )
        write_json(plan_path, plans)
        for start in range(0, len(plans), 60):
            pending = [
                p
                for p in plans[start : start + 60]
                if counts[p["mode"], p["length"]] < QUOTAS[p["mode"]][p["length"]]
            ]
            if not pending:
                continue
            batch_name = f"batch-extra{attempt}-{start // 60:03}"
            source_plan = run / batch_name / "generation/source-plan.json"
            if source_plan.exists():
                pending = json.loads(source_plan.read_text())["plans"]
            for row, meta, review in run_batch(run / batch_name, pending):
                if meta["id"] in seen_ids:
                    continue
                key = (row["mode"], meta["length"])
                if counts[key] >= QUOTAS[key[0]][key[1]]:
                    continue
                index = LeakageIndex(texts)
                if index.matches(row["input"]) or index.matches(row["output"]):
                    exclusions.append({"id": meta["id"], "reason": "new_to_new_overlap"})
                    continue
                retained.append({"row": row, "meta": meta, "review": review, "batch": batch_name})
                counts[key] += 1
                seen_ids.add(meta["id"])
                texts.extend([row["input"], row["output"]])
            write_json(run / "retained.json", retained)
            write_json(run / "overlap-exclusions.json", exclusions)
            print(
                json.dumps({"retained": len(retained), "target": 500, "topup": attempt}), flush=True
            )
            if len(retained) == 500:
                break
    validate_selection(retained)
    write_json(
        run / "final-overlap-check.json",
        {
            "rows": len(retained),
            "new_to_new_overlap": 0,
            "excluded": exclusions,
            "rows_sha256": digest([e["row"] for e in retained]),
        },
    )


def replace_dataset(prepared, current):
    backup = current.with_name(current.name + "-before-boost")
    incoming = current.with_name(current.name + "-incoming-boost")
    if backup.exists():
        raise ValueError("Resolve the saved dataset backup before another replacement")
    if incoming.exists():
        shutil.rmtree(incoming)
    shutil.copytree(prepared, incoming)
    os.replace(current, backup)
    try:
        os.replace(incoming, current)
    except BaseException:
        os.replace(backup, current)
        raise
    shutil.rmtree(backup)


def preserve_batch_evidence(batch, destination):
    """Keep unfinished attempts for audit; only exact verified rows can enter selection."""
    if not (batch / "generation/source-plan.json").exists():
        return
    complete = (batch / "generation/supporting.json").exists()
    names = [
        "summary.json",
        "sol-reviews.json",
        "adjudication/sol-reviews.json",
        "generation/source-plan.json",
        "generation/pipeline.yaml",
        "generation/rubric.yaml",
        "generation/candidates.jsonl",
        "generation/scored-supporting.json",
        "generation/rejected.json",
    ]
    if not complete:
        names.append("generation/runs/author/stages/02_generate/raw_responses.jsonl")
        write_json(
            destination / "incomplete-excluded.json",
            {"eligible_for_selection": False, "reason": "generation_export_incomplete"},
        )
    sources = [batch / name for name in names]
    sources.extend(batch.glob("**/sol-calls/*.json"))
    for source in sources:
        if source.exists():
            target = destination / source.relative_to(batch)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def apply_boost(run):
    """Audit a complete append in staging before updating the existing local dataset."""
    from arka.pipeline.output import OutputWriter

    from .benchmark import COMMON_SYSTEM, Case
    from .reviewed_publication import audit_publication

    run = run.resolve()
    spec = json.loads((run / "spec.json").read_text())
    namespace = spec.get("evidence_namespace", "boost")
    if not re.fullmatch(r"[a-z0-9-]+", namespace):
        raise ValueError("Invalid evidence namespace")
    evidence_name = f"evidence/{namespace}"
    review_name = f"reviews/sol-{namespace}.json"
    backup = CURRENT.with_name(CURRENT.name + "-before-boost")
    if not CURRENT.exists() and backup.exists():
        os.replace(backup, CURRENT)
    existing = CURRENT / evidence_name / "spec.json"
    if existing.exists() and existing.read_bytes() == (run / "spec.json").read_bytes():
        audit_publication(CURRENT)
        if backup.exists():
            shutil.rmtree(backup)
        print(json.dumps({"already_applied": True}))
        return
    if existing.exists():
        raise ValueError("Existing generation evidence must not be overwritten")
    entries = json.loads((run / "retained.json").read_text())
    validate_selection(entries, quotas=spec.get("quotas", QUOTAS))
    if namespace == "english-critical":
        from .english_critical import verify_direction_balance

        verify_direction_balance(entries, json.loads((run / "direction-balance.json").read_text()))
    if (
        "identity_target" in spec
        and sum(e["meta"]["identity"] for e in entries) != spec["identity_target"]
    ):
        raise ValueError("Identity quota is incomplete")
    verify_overlap_report(
        [e["row"] for e in entries], json.loads((run / "final-overlap-check.json").read_text())
    )
    verify_holdouts(
        {name: (CURRENT / name).read_bytes() for name in spec["base_files"]}, spec["base_files"]
    )
    staging = run / "prepared"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(CURRENT, staging)
    evidence = staging / evidence_name
    evidence.mkdir(parents=True, exist_ok=True)
    support = json.loads((staging / "supporting.json").read_text())
    blocked = protected_index()
    native, reviews, batch_cache = {}, [], {}
    for entry in entries:
        row, original, review = entry["row"], entry["meta"], entry["review"]
        meta = {k: v for k, v in original.items() if k != "opus_review"}
        batch = run / entry["batch"]
        if entry["batch"] not in batch_cache:
            stages = batch / "generation/runs/author/stages/03_label_quality"
            scored = {
                r.id: r.model_dump(mode="json")
                for name in ["data.parquet", "dropped.parquet"]
                if (stages / name).exists()
                for r in OutputWriter().read_parquet(stages / name)
            }
            batch_cache[entry["batch"]] = scored
        native[meta["id"]] = batch_cache[entry["batch"]][meta["record_id"]]
        meta["native_exports"] = (
            f"{evidence_name}/batches/{batch.name}/generation/native-export-hashes.json"
        )
        exports = [
            digest(json.loads(line))
            for line in (batch / "generation/native-output.jsonl").read_text().splitlines()
        ]
        write_json(staging / meta["native_exports"], exports)
        if meta["native_accepted"] != (digest(native[meta["id"]]["payload"]) in exports):
            raise ValueError("Native final export membership differs")
        meta["verification_calls"] = []
        expected = [review]
        if meta.get("adjudication_review"):
            expected.append(meta["adjudication_review"])
        for attempt, verdict in enumerate(expected):
            directory = batch / ("sol-calls" if attempt == 0 else "adjudication/sol-calls")
            found = []
            for path in sorted(directory.glob("*.json")):
                record = json.loads(path.read_text())
                if record.get("status") != "completed":
                    continue
                packet = json.loads(record["request"]["messages"][1]["content"])
                if len(packet) != 1 or packet[0]["id"] != meta["id"]:
                    continue
                try:
                    actual = verify_saved_review(row, meta, record)
                except (ValueError, KeyError, TypeError):
                    continue  # Incomplete/schema-rejected attempts remain in batch evidence.
                if actual == verdict:
                    found.append(path)
            if len(found) != 1:
                raise ValueError(f"Expected one exact saved verification call: {meta['id']}")
            name = f"{evidence_name}/sol-calls/{meta['id']}-{attempt + 1}.json"
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found[0], destination)
            meta["verification_calls"].append(name)
        if boost_issues(meta, row, blocked):
            raise ValueError(f"Addition failed final checks: {meta['id']}")
        meta.update(
            data_review=review | {"report": review_name},
            native_evidence=f"{evidence_name}/native-records.json",
            overlap_evidence=f"{evidence_name}/final-overlap-check.json",
            generation_plan=f"{evidence_name}/batches/{batch.name}/generation/source-plan.json",
            human_reviewed=False,
            review_status="sol_verified_machine",
        )
        case = Case(id=meta["id"], **row, context=meta["context"], language=meta["language"])
        guide = {
            "system": COMMON_SYSTEM + MODES[case.mode],
            "user": case.user_message(),
            "assistant": case.output,
        }
        for kind, value in [("data", row), ("sft", guide)]:
            with (staging / f"{kind}/train.jsonl").open("a") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        support.append(meta)
        reviews.append(review)
    # Retain generation plans, decisions, rejected candidates and every review attempt in Git.
    for name in [
        "spec.json",
        "plans.json",
        "overlap-exclusions.json",
        "final-overlap-check.json",
        "direction-balance.json",
        "pre-balance-retained.json",
        "before-quality-backfill-retained.json",
        "quality-backfills.json",
        *[p.name for p in run.glob("extra-plans-*.json")],
    ]:
        if not (run / name).exists():
            continue
        shutil.copyfile(run / name, evidence / name)
    for batch in sorted(run.glob("batch-*")):
        preserve_batch_evidence(batch, evidence / "batches" / batch.name)
    write_json(evidence / "native-records.json", native)
    write_json(staging / "supporting.json", support)
    write_json(
        staging / review_name,
        {
            "reviewer_model": SOL,
            "human_reviewed": False,
            "reviews": reviews,
            "data_policy": namespace,
            "source_spec": f"{evidence_name}/spec.json",
        },
    )
    quality = json.loads((staging / "quality.json").read_text())
    if "excluded_rows" in quality:
        quality["historical_selection"] = {
            name: quality.pop(name) for name in ["excluded_rows", "unused_passing_candidates"]
        }
    quality.setdefault("expansion_history", []).append(
        {
            "previous_source_splits": quality["splits"],
            "previous_first_reviews": quality.get("expansion_first_reviews"),
            "new_namespace": namespace,
        }
    )
    expansion_reviews = [
        review
        for batch in sorted(run.glob("batch-*"))
        if (batch / "sol-reviews.json").exists()
        for review in json.loads((batch / "sol-reviews.json").read_text())["reviews"]
    ]
    quality["expansion_first_reviews"] = dict(Counter(v["verdict"] for v in expansion_reviews))
    pairs = {
        s: [json.loads(line) for line in (staging / f"data/{s}.jsonl").read_text().splitlines()]
        for s in ["train", "dev", "validation"]
    }
    long = [m["tokens"]["input_tokens"] for m in support if m["length"] == "long"]
    quality.update(
        status="model_verified_current_data",
        splits={s: len(rows) for s, rows in pairs.items()},
        by_mode={s: dict(Counter(r["mode"] for r in rows)) for s, rows in pairs.items()},
        independent_targets={
            s: len({r["output"].strip().casefold() for r in rows}) for s, rows in pairs.items()
        },
        source_groups={
            s: len({m["source_group"] for m in support if m["split"] == s}) for s in pairs
        },
        languages=dict(Counter(m["language"] for m in support)),
        long_pairs=len(long),
        long_input_token_range=[min(long), max(long)],
        identity_by_length=dict(Counter(m["length"] for m in support if m.get("identity"))),
        max_chat_tokens=max(m["tokens"]["chat_tokens"] for m in support),
        max_reserved_inference_tokens=max(m["tokens"]["prompt_tokens"] + 2048 for m in support),
        all_exact_rows_opus_pass=False,
        all_exact_rows_review_pass=True,
        new_training_pairs=len(entries),
        new_review_model=SOL,
        new_native_strong=sum(strong_decisions(e["meta"]) for e in entries),
        new_native_uncertain_two_sol_passes=sum(not strong_decisions(e["meta"]) for e in entries),
        new_quotas=spec.get("quotas", QUOTAS),
        holdouts_byte_identical=True,
    )
    write_json(staging / "quality.json", quality)
    (staging / "README.md").write_text(
        f"# Current rewriter training sources\n\n{quality['splits']['train']:,} source training pairs, 100 dev and 84 frozen validation pairs. This append adds {len(entries)} independently generated Arka episodes. All earlier training rows remain unchanged; dev and validation source bytes are unchanged. The active public export selects English only.\n\nNew pairs use Luna generation and six Luna Decisions checks. Complete uncertain native scores require two separate exact-row GPT-6.1 Sol passes. Strong native decisions require one. Confident native rejection cannot be overridden. Each verification checks the input, output and pair. These are model judgments; human calibration remains pending.\n\nHugging Face contains only public Parquet splits, a dataset card and licenses. Code, notebook and exact review evidence stay in Git. Use the single notebook at `../../../notebooks/rewriter_train_l4.ipynb`.\n"
    )
    write_json(
        staging / "manifest.json",
        {
            "role": "current-reviewed-data",
            "files": {
                str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(staging.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    audit_publication(staging)
    replace_dataset(staging, CURRENT)
    print(
        json.dumps({"applied": len(entries), "splits": quality["splits"], "quality": quality}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--multiplier", type=float, default=3)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply a fully verified append to the same local dataset",
    )
    parser.add_argument(
        "--finish",
        action="store_true",
        help="Check new-to-new overlap and fill any remaining quotas",
    )
    args = parser.parse_args()
    if args.finish:
        finish_boost(args.run)
        return
    if args.apply:
        apply_boost(args.run)
        return
    plans = focused_plans(args.multiplier)
    if args.pilot:
        plans = plans[: args.pilot]
    args.run.mkdir(parents=True, exist_ok=True)
    spec = {
        "base_files": {
            f"{kind}/{s}.jsonl": hashlib.sha256(
                (CURRENT / f"{kind}/{s}.jsonl").read_bytes()
            ).hexdigest()
            for kind in ["data", "sft"]
            for s in ["train", "dev", "validation"]
        },
        "plans_sha256": digest(plans),
        "rubric_sha256": digest(focused_rubric()),
        "quotas": QUOTAS,
        "native_strong_score": 0.8,
        "native_strong_confidence": 0.2,
        "native_veto_score": 0.1,
        "native_veto_confidence": 0.8,
        "uncertain_requires_two_sol_passes": True,
        "verifier": SOL,
    }
    if (args.run / "spec.json").exists():
        if json.loads((args.run / "spec.json").read_text()) != spec:
            raise ValueError("Base dataset or generation policy changed; use a new run path")
    else:
        write_json(args.run / "spec.json", spec)
    write_json(args.run / "plans.json", plans)
    retained = []
    counts = Counter()
    seen_targets = set()
    for start in range(0, len(plans), args.batch_size):
        pending = [
            p
            for p in plans[start : start + args.batch_size]
            if counts[p["mode"], p["length"]] < QUOTAS[p["mode"]][p["length"]]
        ]
        if not pending:
            continue
        selected = run_batch(args.run / f"batch-{start // args.batch_size:03}", pending)
        for row, meta, review in selected:
            key = (row["mode"], meta["length"])
            target = " ".join(row["output"].casefold().split())
            if counts[key] < QUOTAS[key[0]][key[1]] and target not in seen_targets:
                retained.append(
                    {
                        "row": row,
                        "meta": meta,
                        "review": review,
                        "batch": f"batch-{start // args.batch_size:03}",
                    }
                )
                counts[key] += 1
                seen_targets.add(target)
        write_json(args.run / "retained.json", retained)
        print(
            json.dumps(
                {
                    "retained": len(retained),
                    "quota": 500,
                    "by_mode_length": {f"{m}:{l}": v for (m, l), v in counts.items()},
                }
            ),
            flush=True,
        )
        if len(retained) == 500:
            break
    print(
        json.dumps({"retained": len(retained), "target": 500, "pilot": bool(args.pilot)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
