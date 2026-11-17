"""Active release auditor verifying local reviewed data, supplements, and exact-row reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from .arka_pipeline import DIMENSIONS
from .benchmark import COMMON_SYSTEM, MODES, Case
from .colab_training import FROZEN_VALIDATION
from .evaluation import digest
from .reviewed_release import load_reviews

PREFIX = "reviewed-v2"
REPO = "jayshah5696/finetuning-labs-rewriter-training"


def plaintext_for_secret_scan(data):
    """Keep raw evidence unchanged; exclude only provider ciphertext from heuristic scans."""
    try:
        document = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return data
    if isinstance(document, dict) and isinstance(document.get("result"), dict):
        for choice in document["result"].get("choices", []):
            for detail in choice.get("message", {}).get("reasoning_details", []) or []:
                if (
                    detail.get("type") == "reasoning.encrypted"
                    and detail.get("format") == "openai-responses-v1"
                    and isinstance(detail.get("data"), str)
                    and detail["data"].startswith("gAAAA")
                ):
                    detail["data"] = "[provider ciphertext]"
    return json.dumps(document, ensure_ascii=False).encode()


def require_script_review(review, script: bytes, notebook: bytes):
    if (
        review.get("reviewer_model") != "gpt-6.1-sol"
        or review.get("verdict") != "pass"
        or review.get("script_sha256") != hashlib.sha256(script).hexdigest()
        or review.get("notebook_sha256") != hashlib.sha256(notebook).hexdigest()
    ):
        raise ValueError("Training script and notebook need an exact passing GPT-6.1 Sol review")


def audit_publication(root: Path, private_needles=()) -> dict[str, bytes]:
    """Verify data provenance without requiring retired code from historical manifests."""
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("role") == "local_data_selection_only":
        raise ValueError("Local data selection has no approved runtime and cannot be published")
    contents = {}
    for name, expected in manifest["files"].items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("unsafe publication path")
        # Historical manifests also index code. Git history retains those artifacts;
        # data integrity and row reviews do not depend on that retired workflow.
        if relative.parts[0] in {"notebooks", "runtime"}:
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"publication hash differs: {name}")
        contents[name] = data
    contents["manifest.json"] = (root / "manifest.json").read_bytes()
    if manifest.get("role") == "reviewed-v3-1000":
        for name, expected in FROZEN_VALIDATION.items():
            if hashlib.sha256(contents[name]).hexdigest() != expected:
                raise ValueError("Frozen v2 validation bytes changed")
    if manifest.get("role") == "current-reviewed-data":
        from .boost_data import verify_holdouts

        freeze = json.loads(
            (Path(__file__).resolve().parents[3] / "data/rewriter/holdout-freeze.json").read_text()
        )
        verify_holdouts(contents, freeze)
    if "validation-bundle/cases.jsonl" in contents and "data/validation.jsonl" in contents:
        bundle_lines = [
            line.strip()
            for line in contents["validation-bundle/cases.jsonl"].splitlines()
            if line.strip()
        ]
        val_lines = [
            line.strip() for line in contents["data/validation.jsonl"].splitlines() if line.strip()
        ]
        if len(bundle_lines) != len(val_lines):
            raise ValueError(
                f"validation-bundle/cases.jsonl count ({len(bundle_lines)}) differs from data/validation.jsonl ({len(val_lines)})"
            )
        for idx, (b_line, v_line) in enumerate(zip(bundle_lines, val_lines, strict=True)):
            b_row = json.loads(b_line)
            v_row = json.loads(v_line)
            if digest(b_row) != digest(v_row) or b_line != v_line:
                raise ValueError(
                    f"validation-bundle/cases.jsonl row {idx} differs from data/validation.jsonl row-for-row"
                )
    for data in contents.values():
        if (
            re.search(rb"sk-or-v1-[A-Za-z0-9_-]{20,}", data)
            or re.search(rb"hf_[A-Za-z0-9]{25,}", plaintext_for_secret_scan(data))
            or any(needle and needle in data for needle in private_needles)
        ):
            raise ValueError("private content in publication")
    required = {
        "supporting.json",
        "quality.json",
        "README.md",
        "DATA_LICENSE.md",
        *(
            f"{kind}/{split}.jsonl"
            for kind in ["data", "sft"]
            for split in ["train", "dev", "validation"]
        ),
    }
    if not required.issubset(contents):
        raise ValueError("publication is missing required data or source notices")
    support = json.loads(contents["supporting.json"])
    quality = json.loads(contents["quality.json"])

    def data_review(item):
        return item.get("data_review") or item["opus_review"]

    review_names = {data_review(m)["report"] for m in support}
    if not review_names.issubset(contents):
        raise ValueError("missing exact review report")
    reviews = load_reviews([root / name for name in sorted(review_names)])
    review_documents = {name: json.loads(contents[name]) for name in review_names}
    plan_documents = {
        name: json.loads(contents[name])["plans"]
        for name in {m["generation_plan"] for m in support if "generation_plan" in m}
    }
    native_documents = {
        name: json.loads(contents[name])
        for name in {m["native_evidence"] for m in support if "data_review" in m}
    }
    from .training import LeakageIndex, normalized, validation_index

    train_texts = [
        json.loads(line)[k]
        for line in contents["data/train.jsonl"].splitlines()
        for k in ["input", "output"]
    ]
    dev_texts = [
        json.loads(line)[k]
        for line in contents["data/dev.jsonl"].splitlines()
        for k in ["input", "output"]
    ]
    val_texts = [
        json.loads(line)[k]
        for line in contents["data/validation.jsonl"].splitlines()
        for k in ["input", "output"]
    ]
    val_frozen_texts = list(validation_index().texts)
    heldout_by_split = (
        {
            "train": LeakageIndex([*val_frozen_texts, *dev_texts, *val_texts]),
            "dev": LeakageIndex([*train_texts, *val_texts]),
            "validation": LeakageIndex([*train_texts, *dev_texts]),
        }
        if native_documents
        else {}
    )
    ids, groups, total = set(), {}, 0
    new_rows, new_meta_rows, new_texts, overlap_groups, expansion_groups = [], [], [], {}, {}
    new_index = LeakageIndex()
    text_to_groups = {}
    for split in ["train", "dev", "validation"]:
        heldout = heldout_by_split.get(split)
        rows = [json.loads(x) for x in contents[f"data/{split}.jsonl"].splitlines()]
        guides = [json.loads(x) for x in contents[f"sft/{split}.jsonl"].splitlines()]
        meta = [m for m in support if m["split"] == split]
        if not (len(rows) == len(guides) == len(meta) == quality["splits"][split]):
            raise ValueError("publication split counts differ")
        total += len(rows)
        for row, guide, item in zip(rows, guides, meta, strict=True):
            if set(row) != {"input", "output", "mode", "checks"}:
                raise ValueError("publication row schema differs")
            case = Case(id=item["id"], **row, context=item["context"], language=item["language"])
            expected_guide = {
                "system": COMMON_SYSTEM + MODES[case.mode],
                "user": case.user_message(),
                "assistant": case.output,
            }
            if guide != expected_guide or digest(row) != item["row_sha256"]:
                raise ValueError("publication supervised text differs from verified row")
            if item["id"] in ids:
                raise ValueError("duplicate publication identity")
            ids.add(item["id"])
            previous = groups.setdefault(item["source_group"], split)
            if previous != split:
                raise ValueError("publication source group crosses splits")
            review = reviews.get((item["id"], digest(row)))
            saved = {k: v for k, v in data_review(item).items() if k != "report"}
            actual = {k: v for k, v in (review or {}).items() if k != "report"}
            if not review or review["verdict"] != "pass" or saved != actual:
                raise ValueError("publication row lacks its exact passing review")
            if item["mechanical_issues"] or item["human_reviewed"] is not False:
                raise ValueError("publication has unresolved checks or false human labels")
            if set(item["decisions"]) != {"labeling_engine." + k for k in DIMENSIONS}:
                raise ValueError("publication is missing native decision dimensions")
            if "data_review" in item:
                from .boost_data import SOL, boost_issues, verified_pair, verify_saved_review
                from .diverse_expansion import is_candidate_admitted

                report_name = item["data_review"]["report"]
                report = review_documents[report_name]
                if report.get("reviewer_model") != SOL:
                    raise ValueError("Unexpected data verification model")
                if "critical_checks" in item or report.get("data_policy") == "english-critical":
                    plans = plan_documents.get(item.get("generation_plan"), [])
                    matching = [p for p in plans if p["id"] == item["id"]]
                    if len(matching) != 1 or any(item.get(k) != v for k, v in matching[0].items()):
                        raise ValueError("Generation plan differs from critical-check metadata")
                    if report.get("data_policy") == "english-critical" and (
                        item["language"] != "en" or "critical_checks" not in item
                    ):
                        raise ValueError("English generation plan is incomplete")
                paths = item.get("verification_calls", [])
                if len(paths) not in {1, 2} or len(set(paths)) != len(paths):
                    raise ValueError("Missing distinct data verification calls")
                records = [json.loads(contents[name]) for name in paths]
                raw_reviews = [verify_saved_review(row, item, record) for record in records]
                second = raw_reviews[1] if len(raw_reviews) == 2 else None
                if raw_reviews[0] != saved or second != item.get("adjudication_review"):
                    raise ValueError("Saved verification review differs from its actual call")
                if len(records) == 2 and (
                    not records[0]["result"].get("id")
                    or records[0]["result"]["id"] == records[1]["result"].get("id")
                ):
                    raise ValueError("Adjudication must use a distinct provider call")
                is_diverse = "family" in item or item.get("source_group", "").startswith(
                    ("expansion-", "contrast-")
                )
                if is_diverse:
                    r1_prov = dict(
                        raw_reviews[0],
                        round=1,
                        call_id=records[0]["result"].get("id"),
                        request_sha256=records[0].get("request_sha256"),
                        response_sha256=records[0].get("result_sha256"),
                    )
                    r2_prov = (
                        dict(
                            second,
                            round=2,
                            call_id=records[1]["result"].get("id"),
                            request_sha256=records[1].get("request_sha256"),
                            response_sha256=records[1].get("result_sha256"),
                        )
                        if second is not None and len(records) == 2
                        else second
                    )
                    is_admitted = is_candidate_admitted(
                        row, item, r1_prov, sol_review_round2=r2_prov
                    )
                else:
                    is_admitted = verified_pair(row, item, raw_reviews[0], second_review=second)
                if not is_admitted or boost_issues(item, row, heldout):
                    raise ValueError("New data lacks completed fidelity verification")
                native = native_documents[item["native_evidence"]][item["id"]]
                exported = json.loads(contents[item["native_exports"]])
                if item["native_accepted"] != (digest(native["payload"]) in exported):
                    raise ValueError("Native final export membership differs")
                if (
                    native["id"] != item["record_id"]
                    or native["payload"]["instruction"] != case.user_message()
                    or native["payload"]["response"] != row["output"]
                    or native["scores"]["decisions"] != item["decisions"]
                    or native["scores"]["judge_model"] != "openai/gpt-6-luna-decisions"
                ):
                    raise ValueError("Native Arka decisions differ from the actual pair")
                for field in ["output", "input"]:
                    for m_text in new_index.matching_texts(row[field]):
                        for g in text_to_groups.get(m_text, ()):
                            if g != item["source_group"]:
                                raise ValueError("New-to-new text overlap in publication")
                same_group_rows = [
                    (prev_r, prev_m)
                    for prev_r, prev_m in zip(new_rows, new_meta_rows, strict=True)
                    if prev_m["source_group"] == item["source_group"]
                ]

                for prev_r, prev_m in same_group_rows:
                    if (
                        prev_m["mode"],
                        (prev_m.get("context") or {}).get("level"),
                    ) == (item["mode"], (item.get("context") or {}).get("level")):
                        raise ValueError("Duplicate mode/level in same source group")
                    if prev_r["input"] != row["input"]:
                        raise ValueError("Different input in same-input contrast group")

                norm_in = normalized(row["input"])
                norm_out = normalized(row["output"])
                new_index.add(row["input"])
                new_index.add(row["output"])
                text_to_groups.setdefault(norm_in, set()).add(item["source_group"])
                text_to_groups.setdefault(norm_out, set()).add(item["source_group"])
                new_rows.append(row)
                new_meta_rows.append(item)
                new_texts.extend([row["input"], row["output"]])
                overlap_name = item.get(
                    "overlap_evidence", "evidence/boost/final-overlap-check.json"
                )
                overlap_groups.setdefault(overlap_name, []).append(row)
                expansion_groups.setdefault(report_name, []).append(
                    {"row": row, "meta": item, "review": saved}
                )
    if new_rows:
        from .boost_data import verify_overlap_report

        for name, rows in overlap_groups.items():
            verify_overlap_report(rows, json.loads(contents[name]))
        from .boost_data import validate_selection, verify_holdouts

        for name, entries in expansion_groups.items():
            report = review_documents[name]
            if "source_spec" not in report:
                continue
            spec = json.loads(contents[report["source_spec"]])
            validate_selection(entries, quotas=spec["quotas"])
            if report.get("data_policy") == "english-critical":
                from .english_critical import verify_direction_balance

                balance_path = str(Path(report["source_spec"]).with_name("direction-balance.json"))
                verify_direction_balance(entries, json.loads(contents[balance_path]))
            if (
                "identity_target" in spec
                and sum(e["meta"]["identity"] for e in entries) != spec["identity_target"]
            ):
                raise ValueError("Published identity quota differs from source plan")
            if "base_counts" in spec:
                parent = {}
                for filename in spec["base_files"]:
                    data = contents[filename]
                    for split_name in ["train", "dev", "validation"]:
                        if (
                            filename.endswith(f"/{split_name}.jsonl")
                            and split_name in spec["base_counts"]
                        ):
                            data = b"".join(
                                data.splitlines(keepends=True)[: spec["base_counts"][split_name]]
                            )
                    parent[filename] = data
                verify_holdouts(parent, spec["base_files"])
    if total != len(support) or Counter(x["split"] for x in support) != quality["splits"]:
        raise ValueError("publication contains foreign split metadata")
    return contents


def main():
    from huggingface_hub import get_token

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    if not args.prepare_only:
        raise SystemExit(
            "Legacy artifact upload retired. Use python -m finetuning_labs.rewriter.hf_data --publish; code and audits stay in Git."
        )
    if not re.fullmatch(r"[a-z0-9-]+", args.prefix):
        raise ValueError("Invalid version prefix")
    prefix = args.prefix
    project = Path(__file__).resolve().parents[3]
    personal = project / "data/rewriter/personal/calibration-10"
    needles = [x.encode() for x in [os.environ.get("OPENROUTER_API_KEY"), get_token()] if x]
    if personal.exists():
        needles.extend(
            json.loads(x)["input"].encode()
            for x in (personal / "cases.jsonl").read_text().splitlines()
        )
        needles.extend(
            x["source_id"].encode()
            for x in json.loads((personal / "supporting/cases.json").read_text())
        )
    verified = audit_publication(args.release, needles)
    contents = {f"{prefix}/{name}": data for name, data in verified.items()}
    contents["README.md"] = verified["README.md"]
    if args.prepare_only:
        print(json.dumps({"verified_files": len(contents), "prefix": prefix, "repo": args.repo}))
        return


if __name__ == "__main__":
    main()
