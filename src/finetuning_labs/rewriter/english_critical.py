"""Generate English fidelity examples through Arka; preserve exact native and Sol evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

QUOTAS = {
    "/slack": {"short": 20, "medium": 20, "long": 10},
    "/transcribe": {"short": 100, "medium": 80, "long": 20},
    "/prompt": {"short": 50, "medium": 60, "long": 40},
    "/polish": {"short": 30, "medium": 40, "long": 30},
}
NAMES = [
    "Amaya",
    "Quentin",
    "Tess",
    "Orson",
    "Leonie",
    "Bram",
    "Celeste",
    "Idris",
    "Nadia",
    "Emmett",
    "Saskia",
    "Jasper",
    "Mina",
    "Otis",
    "Soraya",
    "Ellis",
]
FAMILIES = {
    "/slack": ["recipient_exclusion", "negation", "conditional_fallback", "long_completeness"],
    "/transcribe": [
        "recipient_exclusion",
        "amount_exclusion",
        "correction_context",
        "verbatim",
        "unfinished",
    ],
    "/prompt": [
        "negation",
        "code_exactly_once",
        "conditional_fallback",
        "recipient_exclusion",
        "long_completeness",
    ],
    "/polish": ["actor_causality", "negation", "amount_exclusion", "long_completeness"],
}


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold().replace("don't", "do not")))


def critical_issues(meta, row):
    rules = meta.get("critical_checks", {})
    raw, target = row["input"], row["output"]
    issues = []
    if any(normalized(p) not in normalized(target) for p in rules.get("phrases", [])):
        issues.append("required_relation_changed")
    if any(v not in target for v in rules.get("keep", [])):
        issues.append("required_detail_missing")
    if any(
        re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", target) for v in rules.get("forbid", [])
    ):
        issues.append("superseded_detail_kept")
    if rules.get("ending") and not normalized(target).endswith(normalized(rules["ending"])):
        issues.append("unfinished_thought_completed")
    if rules.get("verbatim_words"):
        filler = rules.get("leading_filler")
        source = (
            re.sub(
                r"\A" + re.escape(filler) + r"(?:[, ]|\s)+", "", raw, count=1, flags=re.IGNORECASE
            )
            if filler
            else raw
        )
        if normalized(source) != normalized(target):
            issues.append("transcribe_words_changed")
    if "code" in rules:

        def blocks(text):
            return re.findall(r"```[^\n]*\n(.*?)\n```", text, flags=re.DOTALL)

        code = rules["code"]
        if blocks(raw) != [code] or blocks(target) != [code] or target.count(code) != 1:
            issues.append("supplied_code_changed_or_duplicated")
    return issues


def _snippet(i):
    field = ["weight", "pages", "minutes", "stock", "distance", "capacity", "credits"][(i // 7) % 7]
    n = 9 + i % 83
    return [
        f"def select_{field}(rows):\n    return [row for row in rows if row.get('{field}', 0) >= {n}]",
        f"def count_{field}(values):\n    counts = {{}}\n    for value in values:\n        counts[value] = counts.get(value, 0) + 1\n    return counts",
        f"records = [{{'{field}': {n}}}, {{'{field}': {n + 7}}}]\nselected = [record['{field}'] for record in records if record['{field}'] > {n + 2}]",
        f"def group_{field}(rows):\n    groups = {{}}\n    for row in rows:\n        groups.setdefault(row['{field}'], []).append(row)\n    return groups",
        f"values = [{n}, {n + 4}, {n}]\nunique = []\nfor value in values:\n    if value not in unique:\n        unique.append(value)",
        f"def read_{field}(record):\n    value = record.get('{field}')\n    if value is None:\n        return {n}\n    return int(value)",
        f"def sort_{field}(rows):\n    return sorted(rows, key=lambda row: row.get('{field}', {n}), reverse=True)",
    ][i % 7]


def plans(multiplier=4, round_number=0, namespace="main"):
    from .boost_data import DOMAINS

    rng = random.Random(291026 + round_number)
    total = sum(math.ceil(n * multiplier) for cells in QUOTAS.values() for n in cells.values())
    identities = set(rng.sample(range(total), total // 4))
    domains = [d.strip() for d in DOMAINS.replace("\n", "").split("|")]
    items = []
    for mode, cells in QUOTAS.items():
        for length, quota in cells.items():
            for i in range(math.ceil(quota * multiplier)):
                serial = len(items) + round_number * 5000
                focus = FAMILIES[mode][i % len(FAMILIES[mode])]
                identity = len(items) in identities
                domain = rng.choice(domains)
                person, other = NAMES[serial % len(NAMES)], NAMES[(serial + 5) % len(NAMES)]
                checks, protected = {}, []
                detail = ""
                if focus == "recipient_exclusion":
                    checks["phrases"] = [f"{person} not {other}"]
                    detail = f"Use a clear recipient instruction with '{person}, not {other}'. Keep that exclusion and recipient exactly."
                elif focus == "amount_exclusion":
                    first, second = (
                        f"{31 + serial % 63}.{serial % 100:02}",
                        f"{101 + serial % 79}.{(serial + 27) % 100:02}",
                    )
                    checks["phrases"] = [f"{first} not {second}"]
                    detail = f"Use an actual cost of ${first}, not ${second}. Keep the correct and explicitly rejected amounts with their negation, not a replacement of the correct amount."
                elif focus == "negation":
                    action = [
                        "rename files",
                        "contact vendors",
                        "install packages",
                        "publish photos",
                        "charge accounts",
                        "open cloud links",
                        "modify records",
                    ][serial % 7]
                    checks["phrases"] = [f"do not {action}"]
                    detail = f"Include the explicit restriction 'do not {action}' and an ordinary allowed action. Preserve the prohibition and its scope exactly."
                elif focus == "correction_context":
                    final, cancelled = f"B{11 + serial % 73}", f"A{101 + serial % 71}"
                    checks = {"keep": [final, person], "forbid": [cancelled]}
                    detail = f"Dictate a real action involving location {cancelled}, actually say that obsolete location first, then clearly correct it to {final}. Keep {person} as a separate named participant and preserve another concrete time or instruction. The target retains the entire resolved request, {final} and {person}, removes {cancelled} and the spoken repair cue. No ambiguity or reported correction story."
                elif focus == "verbatim":
                    checks = {"verbatim_words": True, "leading_filler": "um"}
                    detail = "Start the raw input with exactly one 'um' hesitation, then a natural message containing meaningful 'uh-huh', 'well', 'like' or 'actually'. Remove only the initial um. Preserve every other word in order, including the meaningful affirmation. No other restarts, corrections or spoken punctuation cues. Distribute capitalization and punctuation repairs through the text."
                elif focus == "unfinished":
                    ending = ["we could", "I might", "they can", "we would"][serial % 4]
                    checks.update(ending=ending, verbatim_words=True)
                    detail = f"End an intentionally unfinished conditional thought with exactly '{ending}', with no action after it. Preserve that unfinished ending and the complete earlier context. Never invent 'proceed' or another completion. Use missing punctuation and lowercase starts, with no fillers, restarts or spoken corrections. Preserve every word in its original order."
                elif focus == "code_exactly_once":
                    code = _snippet(serial)
                    checks["code"] = code
                    protected = [code]
                    domain = "a small software utility with supplied code"
                    detail = f"Supply this entire snippet in exactly one fenced code block in both input and target:\n```python\n{code}\n```\nAsk for explanation or review using only the snippet. Preserve the complete code exactly once. The future explainer may give the requested explanation, but must not add a wrapper, a second code version, tests, imports or extra code. Keep the specific explanation question and beginner background."
                elif focus == "conditional_fallback":
                    detail = "Give a concrete requested action that is allowed only if a stated condition holds, and a useful different action otherwise. Preserve the requested object, condition, negation, alternative and any output-only constraint. The fallback is not the primary task."
                elif focus == "actor_causality":
                    detail = "Describe a definite event, its explicit practical cause, and a separate personal reaction or hedge. Keep the cause, reaction, actor and degree of certainty separate. Improve genuinely awkward wording without merging causal clauses or deleting qualifications."
                else:
                    detail = "Use a self-contained ordinary update with several substantive facts and a clear final section. Preserve the entire last assessment, conditions and scope rather than inventing a summary."
                if identity:
                    checks, protected = {}, []
                    detail = "Create an already correct, natural text. The target MUST be byte-identical to the input, including punctuation, quotes, names, lists and paragraph breaks. Do not improve optional style or synonyms."
                brief = f"Write one independent fictional English episode about {domain}. {detail} "
                if length == "long":
                    brief += "Use substantive connected paragraphs, an ordinary quote, a numbered section when natural, and a final paragraph with distinct information. No repeated padding, repeated summaries, stock uncertainty formulas or authoring instructions. Keep facts spread through the whole document. "
                    if mode == "/slack":
                        brief += (
                            "Use a realistic pasted thread of short posts by different speakers. "
                        )
                if (
                    mode == "/transcribe"
                    and focus not in {"correction_context", "verbatim", "unfinished"}
                    and not identity
                ):
                    checks["verbatim_words"] = True
                    brief += "Preserve all spoken words in order. Use lowercase starts and missing punctuation as the noise. No spoken punctuation cues or extra fillers, and no paraphrase. "
                case_id = f"arka-english-{namespace}-{round_number}-{mode[1:]}-{length}-{i + 1:04}"
                items.append(
                    {
                        "id": case_id,
                        "source_group": case_id,
                        "split": "train",
                        "mode": mode,
                        "language": "en",
                        "length": length,
                        "identity": identity,
                        "context": {},
                        "domain": domain,
                        "task_family": focus,
                        "focus": focus,
                        "protected_literals": protected,
                        "critical_checks": checks,
                        "brief": brief,
                    }
                )
    rng.shuffle(items)
    return items


def saved_batch_plans(batch, pending):
    path = batch / "generation/source-plan.json"
    return json.loads(path.read_text())["plans"] if path.exists() else pending


def fidelity_long_plans():
    """Add fresh long requests whose organization changes preserve every substantive word."""
    pool = [
        p
        for p in plans(round_number=4, namespace="fidelity")
        if p["mode"] == "/prompt" and p["length"] == "long" and not p["identity"]
    ][:96]
    for plan in pool:
        plan["critical_checks"]["verbatim_words"] = True
        plan["brief"] += (
            " Make the input a natural rough dictated request with lowercase sentence starts and "
            "missing sentence punctuation throughout. Organize the complete request into readable "
            "paragraphs by repairing punctuation, capitalization and paragraph breaks. Preserve "
            "every original word in its original order, including the final section. Do not "
            "compress, paraphrase or add headings. Use numbered items only if their numbers and "
            "words already occur in the input. Preserve supplied quotes and code exactly."
        )
    return pool


def completion_pool(source, attempted, counts, identities):
    return [
        p
        for p in source
        if p["id"] not in attempted
        and counts.get((p["mode"], p["length"]), 0) < QUOTAS[p["mode"]][p["length"]]
        and identities.get(p["identity"], 0) < (125 if p["identity"] else 375)
    ]


def complete_run(run, *, pool_name="completion"):
    """Batch the remaining open cells together, preserving every prior attempt and verdict."""
    from .boost_data import CURRENT, run_batch, validate_selection, verify_holdouts
    from .evaluation import digest, write_json
    from .training import LeakageIndex

    if not re.fullmatch(r"[a-z0-9-]+", pool_name):
        raise ValueError("Invalid completion pool name")
    spec = json.loads((run / "spec.json").read_text())
    verify_holdouts(
        {name: (CURRENT / name).read_bytes() for name in spec["base_files"]}, spec["base_files"]
    )
    retained = json.loads((run / "retained.json").read_text())
    counts = Counter((e["row"]["mode"], e["meta"]["length"]) for e in retained)
    identities = Counter(e["meta"]["identity"] for e in retained)
    ids = {e["meta"]["id"] for e in retained}
    pool_path = run / f"extra-plans-{pool_name}.json"
    if not pool_path.exists():
        if pool_name == "fidelity-long":
            write_json(pool_path, fidelity_long_plans())
        elif pool_name != "completion":
            raise ValueError("A named supplemental source pool must already exist")
        else:
            attempted = {
                p["id"]
                for path in run.glob("batch-*/generation/source-plan.json")
                for p in json.loads(path.read_text())["plans"]
            }
            source = [p for round_number in range(4) for p in plans(round_number=round_number)]
            write_json(pool_path, completion_pool(source, attempted, counts, identities))
    pool = json.loads(pool_path.read_text())
    for start in range(0, len(pool), 96):
        batch_name = f"batch-{pool_name}-{start // 96:03}"
        batch = run / batch_name
        for row, meta, review in run_batch(
            batch, saved_batch_plans(batch, pool[start : start + 96])
        ):
            key = row["mode"], meta["length"]
            if (
                meta["id"] in ids
                or counts[key] >= QUOTAS[key[0]][key[1]]
                or identities[meta["identity"]] >= (125 if meta["identity"] else 375)
            ):
                continue
            blocked = LeakageIndex([e["row"][k] for e in retained for k in ["input", "output"]])
            if blocked.matches(row["input"]) or blocked.matches(row["output"]):
                continue
            retained.append({"row": row, "meta": meta, "review": review, "batch": batch_name})
            counts[key] += 1
            identities[meta["identity"]] += 1
            ids.add(meta["id"])
        write_json(run / "retained.json", retained)
        print(json.dumps({"english_retained": len(retained), "target": 500}), flush=True)
        if len(retained) == 500:
            break
    validate_selection(retained, quotas=QUOTAS)
    write_json(
        run / "final-overlap-check.json",
        {
            "rows": len(retained),
            "new_to_new_overlap": 0,
            "rows_sha256": digest([e["row"] for e in retained]),
        },
    )


def reverse_source(source, attempt):
    plan = json.loads(json.dumps(source))
    checks = plan["critical_checks"]
    if plan["focus"] in {"amount_exclusion", "recipient_exclusion"}:
        first, second = checks["phrases"][0].split(" not ")
        checks["phrases"] = [second + " not " + first]
    elif plan["focus"] == "correction_context":
        first, second = checks["keep"][0], checks["forbid"][0]
        checks["keep"][0], checks["forbid"][0] = second, first
    else:
        raise ValueError("Only amount, recipient and correction sources have a reverse direction")
    replacements = {first: second, second: first}
    pattern = "|".join(re.escape(s) for s in sorted(replacements, key=len, reverse=True))
    plan["brief"] = re.sub(pattern, lambda match: replacements[match.group()], plan["brief"])
    plan["brief"] += (
        " Create a fresh independent event with these facts, not a revision of an earlier episode."
    )
    plan["id"] += f"-reverse-{attempt:02}"
    plan["source_group"] = plan["id"]
    plan["contrast_direction"] = "reverse"
    return plan


def reverse_candidates(source, attempt):
    long_prompt = source["mode"] == "/prompt" and source["length"] == "long"
    candidates = [reverse_source(source, attempt * 16 + n) for n in range(16 if long_prompt else 4)]
    if long_prompt:
        domains = [
            "a community library repair plan with supplied meeting notes",
            "a local museum exhibit brief with supplied interview excerpts",
            "a transit accessibility audit with supplied field observations",
            "a volunteer event plan with supplied messages from coordinators",
        ]
        for n, plan in enumerate(candidates):
            domain = domains[(attempt + n) % len(domains)]
            plan["brief"] = plan["brief"].replace(plan["domain"], domain, 1)
            plan["domain"] = domain
            plan["brief"] += (
                " Include substantive source notes alongside the actual request, so the long "
                "input is useful at this length. Preserve all supplied material inside the "
                "rewritten prompt instead of summarizing it or performing the requested work."
            )
    return candidates


def verify_direction_balance(entries, report):
    from .evaluation import digest

    if report.get("selected_rows_sha256") != digest([e["row"] for e in entries]):
        raise ValueError("Direction balance report differs from selected rows")
    amounts, locations, recipients = Counter(), Counter(), Counter()
    for entry in entries:
        meta = entry["meta"]
        if meta["identity"]:
            continue
        if meta["focus"] == "amount_exclusion":
            left, right = meta["critical_checks"]["phrases"][0].split(" not ")
            amounts["larger" if Decimal(left) > Decimal(right) else "smaller"] += 1
        elif meta["focus"] == "correction_context":
            locations[meta["critical_checks"]["keep"][0][0]] += 1
        elif meta["focus"] == "recipient_exclusion":
            recipients[meta.get("contrast_direction", "forward")] += 1
    for counts, directions in [
        (amounts, ["larger", "smaller"]),
        (locations, ["A", "B"]),
        (recipients, ["forward", "reverse"]),
    ]:
        total = sum(counts.values())
        if not total or any(not 0.35 <= counts[d] / total <= 0.65 for d in directions):
            raise ValueError("Source direction balance is incomplete")
    return {
        "amounts": dict(amounts),
        "final_location_prefix": dict(locations),
        "recipient_direction": dict(recipients),
    }


def balance_run(run):
    from .boost_data import CURRENT, run_batch, validate_selection, verify_holdouts
    from .evaluation import digest, write_json
    from .training import LeakageIndex

    run = run.resolve()
    spec = json.loads((run / "spec.json").read_text())
    verify_holdouts(
        {name: (CURRENT / name).read_bytes() for name in spec["base_files"]}, spec["base_files"]
    )
    original_path = run / "pre-balance-retained.json"
    if not original_path.exists():
        original_path.write_bytes((run / "retained.json").read_bytes())
    original = json.loads(original_path.read_text())
    validate_selection(original, quotas=QUOTAS)
    retained = list(original)
    groups = {}
    for i, e in enumerate(original):
        m = e["meta"]
        if not m["identity"] and m["focus"] in {
            "amount_exclusion",
            "correction_context",
            "recipient_exclusion",
        }:
            groups.setdefault((e["row"]["mode"], m["length"], m["focus"]), []).append(i)
    replacements, exclusions, sources, cell_keys = [], [], {}, {}
    for group_index, (key, indices) in enumerate(sorted(groups.items())):
        for index in indices[: len(indices) // 2]:
            old = original[index]
            source_plan = json.loads(
                (run / old["batch"] / "generation/source-plan.json").read_text()
            )
            sources[index] = next(p for p in source_plan["plans"] if p["id"] == old["meta"]["id"])
            cell_keys[index] = key
    index_for_id = {p["id"]: index for index, p in sources.items()}
    resolved = set()
    for attempt in range(8):
        plan_path = run / f"extra-plans-balance-{attempt}.json"
        if plan_path.exists():
            pool = json.loads(plan_path.read_text())
        else:
            pool = [
                candidate
                for index, source in sources.items()
                if index not in resolved
                for candidate in reverse_candidates(source, attempt)
            ]
            write_json(plan_path, pool)
        for start in range(0, len(pool), 96):
            batch_name = f"batch-balance-{attempt}-{start // 96:03}"
            batch = run / batch_name
            candidates = saved_batch_plans(batch, pool[start : start + 96])
            for row, meta, review in run_batch(batch, candidates):
                index = index_for_id[meta["id"].rsplit("-reverse-", 1)[0]]
                if index in resolved:
                    continue
                blocked = LeakageIndex(
                    [
                        e["row"][k]
                        for j, e in enumerate(retained)
                        if j != index
                        for k in ["input", "output"]
                    ]
                )
                if blocked.matches(row["input"]) or blocked.matches(row["output"]):
                    exclusions.append({"id": meta["id"], "reason": "selection_overlap"})
                    continue
                retained[index] = {"row": row, "meta": meta, "review": review, "batch": batch_name}
                resolved.add(index)
                old = original[index]
                replacements.append(
                    {
                        "original_id": old["meta"]["id"],
                        "original_sha256": old["meta"]["row_sha256"],
                        "replacement_id": meta["id"],
                        "replacement_sha256": meta["row_sha256"],
                        "cell": list(cell_keys[index]),
                    }
                )
            print(
                json.dumps({"balanced_replacements": len(replacements), "required": len(sources)}),
                flush=True,
            )
        if len(resolved) == len(sources):
            break
    if len(resolved) != len(sources):
        raise ValueError("Some reverse-direction slots lack passing independent examples")
    validate_selection(retained, quotas=QUOTAS)
    assert sum(e["meta"]["identity"] for e in retained) == 125
    write_json(run / "retained.json", retained)
    balance = {
        "source_selection_sha256": digest([e["row"] for e in original]),
        "selected_rows_sha256": digest([e["row"] for e in retained]),
        "replacements": replacements,
        "overlap_rejections": exclusions,
    }
    balance["counts"] = verify_direction_balance(retained, balance)
    write_json(run / "direction-balance.json", balance)
    write_json(
        run / "final-overlap-check.json",
        {
            "rows": len(retained),
            "new_to_new_overlap": 0,
            "rows_sha256": digest([e["row"] for e in retained]),
        },
    )


def main():
    from .boost_data import CURRENT, SOL, apply_boost, run_batch, validate_selection
    from .evaluation import digest, write_json
    from .training import LeakageIndex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--complete-pool", default="completion")
    args = parser.parse_args()
    if args.apply:
        apply_boost(args.run)
        return
    if args.balance:
        balance_run(args.run)
        return
    if args.complete:
        complete_run(args.run, pool_name=args.complete_pool)
        return
    args.run.mkdir(parents=True, exist_ok=True)
    spec = {
        "evidence_namespace": "english-critical",
        "quotas": QUOTAS,
        "identity_target": 125,
        "base_counts": json.loads((CURRENT / "quality.json").read_text())["splits"],
        "verifier": SOL,
        "generator": "openai/gpt-6-luna",
        "base_files": {
            f"{kind}/{split}.jsonl": hashlib.sha256(
                (CURRENT / f"{kind}/{split}.jsonl").read_bytes()
            ).hexdigest()
            for kind in ["data", "sft"]
            for split in ["train", "dev", "validation"]
        },
        "plan_sha256": digest(plans(namespace="pilot" if args.pilot else "main")),
        "pilot": args.pilot,
    }
    path = args.run / "spec.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Source dataset or English policy changed; use another run path")
    write_json(path, spec)
    retained, texts, counts = [], [], Counter()
    identity_counts = Counter()
    for round_number in range(4):
        pool = plans(round_number=round_number, namespace="pilot" if args.pilot else "main")
        if args.pilot:
            pool = pool[: args.pilot]
        write_json(args.run / f"extra-plans-{round_number}.json", pool)
        for start in range(0, len(pool), 96):
            pending = [
                p
                for p in pool[start : start + 96]
                if counts[p["mode"], p["length"]] < QUOTAS[p["mode"]][p["length"]]
            ]
            if not pending:
                continue
            batch_name = f"batch-r{round_number}-{start // 96:03}"
            pending = saved_batch_plans(args.run / batch_name, pending)
            for row, meta, review in run_batch(args.run / batch_name, pending):
                key = row["mode"], meta["length"]
                if counts[key] >= QUOTAS[key[0]][key[1]]:
                    continue
                if identity_counts[meta["identity"]] >= (125 if meta["identity"] else 375):
                    continue
                blocked = LeakageIndex(texts)
                if blocked.matches(row["input"]) or blocked.matches(row["output"]):
                    continue
                retained.append({"row": row, "meta": meta, "review": review, "batch": batch_name})
                counts[key] += 1
                identity_counts[meta["identity"]] += 1
                texts.extend([row["input"], row["output"]])
            write_json(args.run / "retained.json", retained)
            print(
                json.dumps(
                    {
                        "english_retained": len(retained),
                        "target": 500,
                        "counts": {f"{m}:{l}": n for (m, l), n in counts.items()},
                    }
                ),
                flush=True,
            )
            if len(retained) == 500:
                break
        if args.pilot or len(retained) == 500:
            break
    if not args.pilot:
        validate_selection(retained, quotas=QUOTAS)
        write_json(
            args.run / "final-overlap-check.json",
            {
                "rows": len(retained),
                "new_to_new_overlap": 0,
                "rows_sha256": digest([e["row"] for e in retained]),
            },
        )


if __name__ == "__main__":
    main()
