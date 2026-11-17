"""Current diagnostic script: describe training coverage and repeated wording, without assigning semantic quality."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.release / "data/train.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    metadata = [
        m
        for m in json.loads((args.release / "supporting.json").read_text())
        if m["split"] == "train"
    ]
    pairs = list(zip(rows, metadata, strict=True))
    grams = Counter()
    for row in rows:
        words = re.findall(r"\w+", row["output"].casefold())
        grams.update({" ".join(words[i : i + 8]) for i in range(len(words) - 7)})
    english = [r for r, m in pairs if m["language"] == "en" and not m.get("identity")]
    long = [r for r, m in pairs if m["length"] == "long"]
    report = {
        "training_data_sha256": hashlib.sha256(raw).hexdigest(),
        "train_rows": len(rows),
        "distinct_targets": len({r["output"].strip().casefold() for r in rows}),
        "distinct_new_domains": len(
            {m["domain"] for m in metadata if not m.get("carried_from_v2")}
        ),
        "top_eight_word_sequences_by_documents": sorted(grams.items(), key=lambda x: (-x[1], x[0]))[
            :10
        ],
        "lowercase_start_in_nonidentity_english_inputs": {
            "count": sum(r["input"].lstrip()[:1].islower() for r in english),
            "denominator": len(english),
        },
        "mode_lengths": {
            mode: dict(Counter(m["length"] for r, m in pairs if r["mode"] == mode))
            for mode in ["/slack", "/transcribe", "/prompt", "/polish"]
        },
        "identity_train": sum(bool(m.get("identity")) for m in metadata),
        "long_train": len(long),
        "long_with_numbered_target_sections": sum(
            bool(re.search(r"(?m)^\s*\d+[.)]\s", r["output"])) for r in long
        ),
        "long_with_quoted_target_text": sum(any(c in r["output"] for c in '"“”„«»') for r in long),
        "long_with_paragraph_breaks": sum("\n\n" in r["output"] for r in long),
        "meaning_quality_established_by_this_audit": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
