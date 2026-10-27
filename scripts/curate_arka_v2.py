"""Apply recorded data corrections after Opus review, preserving all earlier artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from finetuning_labs.rewriter.arka_pipeline import candidate_issues, write_verification_pack
from finetuning_labs.rewriter.benchmark import load_bundle
from finetuning_labs.rewriter.evaluation import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def build_pool():
    pool = {}
    for name in [
        "arka-native-v2",
        "arka-native-v2-length-repairs",
        "arka-native-v2-extra-identities",
    ]:
        root = ROOT / "artifacts" / name
        rows = [json.loads(x) for x in (root / "candidates.jsonl").read_text().splitlines()]
        path = root / "scored-supporting.json"
        metadata = json.loads((path if path.exists() else root / "supporting.json").read_text())
        for row, meta in zip(rows, metadata, strict=True):
            if meta["id"] in pool:
                raise ValueError("duplicate source identity")
            pool[meta["id"]] = (row, {**meta, "native_source_run": name})
    support = json.loads((ROOT / "data/rewriter/benchmark-draft/supporting/cases.json").read_text())
    for case, old in zip(load_bundle().cases, support, strict=True):
        row = {k: getattr(case, k) for k in ["input", "output", "mode", "checks"]}
        pool[case.id] = (
            row,
            {
                "id": case.id,
                "split": "validation",
                "source_group": old["source_group"],
                "language": case.language,
                "context": case.context,
                "length": "long" if case.id.startswith("long-") else "short",
                "identity": case.rules.identity,
                "rules": old["rules"],
                "row_sha256": digest(row),
                "mechanical_issues": [],
                "original_benchmark_row_sha256": digest(row),
            },
        )
    return pool


def curate(destination: Path):
    if destination.exists():
        raise FileExistsError(destination)
    pool = build_pool()
    original = {key: digest(row) for key, (row, meta) in pool.items()}
    ledger = []

    def replace(case_id, field, before, after, *, count=1):
        row, _meta = pool[case_id]
        if row[field].count(before) != count:
            raise ValueError(f"{case_id}: correction context changed: {before!r}")
        row[field] = row[field].replace(before, after)
        ledger.append({"id": case_id, "field": field, "before": before, "after": after})

    def set_text(case_id, field, value):
        row, _meta = pool[case_id]
        ledger.append({"id": case_id, "field": field, "before": row[field], "after": value})
        row[field] = value

    replace("tw-profiles-03", "output", "3. Benchmarks.", "3. Benchmarks")
    set_text(
        "polish-06",
        "input",
        "the timeout is 500 milliseconds not five seconds i said five seconds earlier but that was wrong",
    )
    replace(
        "arka-v2-transcribe-21",
        "output",
        "dá para mudar minha aula de direção de terça-feira para quinta-feira",
        "dá pra mudar minha aula de direção de terça pra quinta-feira",
    )
    replace("arka-v2-transcribe-23", "output", '"cache.enabled,"', '"cache.enabled",')
    replace(
        "arka-v2-transcribe-24",
        "output",
        "the field for the date she turned up. The following week, she arrived and the pass had expired.",
        "the field for the date. She turned up the following week, and the pass had expired.",
    )
    replace(
        "arka-v2-transcribe-24", "output", "Fourth, make the locker", "And fourth, make the locker"
    )
    replace(
        "arka-v2-prompt-23",
        "output",
        "Distinguish documented facts from uncertainty, and mark any criterion the materials do not cover as “not established.”",
        "Distinguish documented facts from uncertainty. Say when the supplied information doesn't establish something.",
    )
    pool["arka-v2-prompt-23"][0]["output"] += "\n\nKeep it practical and concise."
    replace(
        "arka-v2-prompt-24",
        "output",
        "Refer to the exercise leader as the facilitator",
        "Do not assume there is a separate crisis office. Refer to the exercise leader as the facilitator",
    )
    replace(
        "arka-v2-prompt-24",
        "output",
        "## Exercise purpose, scope, and boundaries\n\n",
        "## Exercise purpose, scope, assumptions, and boundaries\n\nDescribe the exercise purpose, scope, assumptions and boundaries up front. ",
    )
    replace("arka-v2-prompt-24", "output", " in a later process", "")
    replace(
        "arka-v2-prompt-24", "output", "printed or otherwise provided inject cards", "inject cards"
    )
    pool["arka-v2-prompt-24"][0]["output"] += "\n\nProduce materials we can adapt later."

    for case_id, before, after in [
        ("arka-v2-slack-10", "hey,", "Hey,"),
        ("arka-v2-transcribe-18", "please", "Please"),
    ]:
        for field in ["input", "output"]:
            replace(case_id, field, before, after)
    replace(
        "arka-v2-slack-12",
        "output",
        "MODE: /slack\nLANGUAGE: en\nINPUT:\nmorning all",
        "Mara — 9:12\nMorning all",
    )
    replace(
        "arka-v2-slack-12",
        "output",
        "before we cut the next test pass. i tried",
        "before we cut the next test pass. I tried",
    )
    replace("arka-v2-slack-12", "output", "; there was no obvious loss", "; no obvious loss")
    replace(
        "arka-v2-slack-12", "output", "The second reproducible issue", "The second reproducible one"
    )
    replace("arka-v2-slack-12", "output", "firmware if it's visible", "firmware if it is visible")
    replace(
        "arka-v2-transcribe-08",
        "output",
        "that was just what I could see",
        "it was just what I could see",
    )
    replace(
        "arka-v2-transcribe-08", "output", "but the solar equipment", "but that the solar equipment"
    )
    replace(
        "arka-v2-transcribe-09",
        "input",
        "no domingo não na segunda-feira",
        "no domingo não quer dizer na segunda-feira",
    )
    replace("arka-v2-transcribe-19", "output", "Correction: keep the sentence", "Keep the sentence")
    replace(
        "arka-v2-transcribe-20",
        "output",
        "I have been on the platform team",
        "So I have been on the platform team",
    )
    replace(
        "arka-v2-transcribe-20",
        "output",
        "what onboarding has actually been like. A few things",
        "what onboarding has actually been like because a few things",
    )
    replace(
        "arka-v2-transcribe-20",
        "output",
        "and is production access expected",
        "and whether production access is expected",
    )
    replace(
        "arka-v2-transcribe-20",
        "output",
        "she said, “Answer the comment first, then request whoever owns the area.”",
        "she said answer the comment first, then request whoever owns the area.",
    )
    replace("arka-v2-transcribe-20", "output", "Release Flow", "release flow", count=2)
    replace("arka-v2-transcribe-20", "output", "`make check`", "make check")
    set_text(
        "arka-v2-prompt-05",
        "output",
        "Turn these interview notes into a short article for local library patrons. Keep the quotations distinct and do not invent quotations.\n\nMira: “The garden gives neighbors a reason to meet.”\nLuis: “I learned how to grow tomatoes here.”",
    )
    replace(
        "arka-v2-prompt-08",
        "output",
        "Groceries are often shared, but the split varies by trip",
        "Groceries are usually shared but not always, and the split varies by trip",
    )
    replace(
        "arka-v2-prompt-08",
        "output",
        "field names, formula logic and a short routine for using it.",
        "field names, formula logic, explanations of what each part is for, and a short routine for using it.",
    )
    replace(
        "arka-v2-prompt-12",
        "input",
        "; produce the requirements document request as a clear prompt.",
        ".",
    )
    replace(
        "arka-v2-polish-03",
        "output",
        "and contribute to the neighborhood",
        "and be part of something useful in the neighborhood",
    )
    replace(
        "arka-v2-polish-03",
        "output",
        "afternoon sessions are organized",
        "afternoon sessions are usually organized",
    )
    replace(
        "arka-v2-polish-03",
        "output",
        "I can also volunteer one Saturday morning each month, though I may need to check the date in advance.",
        "I can also volunteer one Saturday morning each month. My schedule is usually steady, though I may need to check the date in advance for Saturdays.",
    )
    replace("arka-v2-polish-08", "input", "the list had 24 artists", "the list had 18 artists")
    replace(
        "arka-v2-polish-08",
        "output",
        "the list included 24 artists",
        "the list included 18 artists",
    )
    replace("arka-v2-polish-08", "output", "“Low Tide / High Street,”", "“Low Tide / High Street”,")
    replace("arka-v2-polish-08", "output", "“Low tide, high street.”", "“Low tide, high street”.")
    replace(
        "arka-v2-polish-12",
        "output",
        "which looked uncomfortable.",
        "which looked a bit uncomfortable for them.",
    )
    replace(
        "arka-v2-polish-15",
        "output",
        "but these comments alone seem too broad to support that.",
        "but that claim seems too broad from these comments alone.",
    )
    replace(
        "arka-v2-polish-17",
        "output",
        "Hey, could you please keep",
        "Hey, I wanted to ask if you could maybe keep",
    )
    set_text(
        "arka-v2-slack-19",
        "input",
        "hey finance the customer asked if they can split this invoice into two payments i haven't seen an approval yet can we offer that or should i wait before replying",
    )
    set_text(
        "arka-v2-slack-19",
        "output",
        "Hey finance, the customer asked if they can split this invoice into two payments. I haven't seen an approval yet. Can we offer that, or should I wait before replying?",
    )
    pool["arka-v2-slack-19"][1]["length"] = "short"

    rows, metadata = [], []
    for case_id, (row, meta) in pool.items():
        meta["previous_row_sha256"] = original[case_id]
        meta["row_sha256"] = digest(row)
        meta["changed_after_review"] = digest(row) != original[case_id]
        meta["mechanical_issues"] = (
            candidate_issues(meta, row) if case_id.startswith("arka-v2-") else []
        )
        rows.append(row)
        metadata.append(meta)
    destination.mkdir(parents=True)
    (destination / "candidates.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    )
    write_json(destination / "supporting.json", metadata)
    write_json(destination / "corrections.json", ledger)
    for split in ["train", "dev", "validation"]:
        selected = [(r, m) for r, m in zip(rows, metadata, strict=True) if m["split"] == split]
        write_verification_pack(
            [x[0] for x in selected], [x[1] for x in selected], destination / f"opus-{split}"
        )
    return {
        "rows": len(rows),
        "changed_rows": sum(x["changed_after_review"] for x in metadata),
        "edits": len(ledger),
    }


if __name__ == "__main__":
    print(json.dumps(curate(ROOT / "artifacts/arka-curated-v2-round3"), indent=2))
