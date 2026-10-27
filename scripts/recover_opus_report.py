"""Recover the omitted review array from the reviewer's saved tool command.

This parses literals only. It does not execute the command or invent verdicts.
Keep the incomplete original report and the original command as evidence.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "artifacts/opus-reviews"
    command = (folder / "round3-train-dev-command.txt").read_text()
    code = command.split("<<'EOF'\n", 1)[1].split("\nEOF\n", 1)[0]
    tree = ast.parse(code)
    declarations = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    passed = ast.literal_eval(declarations["P"])

    class ExpandPass(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id != "P":
                raise ValueError("Unexpected variable in reviewer literals")
            return ast.Tuple(elts=[ast.Constant(x) for x in passed], ctx=ast.Load())

    critiques = ast.literal_eval(ExpandPass().visit(declarations["C"]))
    report = json.loads((folder / "round3-train-dev.json").read_text())
    assert "reviews" not in report
    reviews = []
    packets = ROOT / "artifacts/arka-curated-v2-round3/opus-pending-train-dev"
    for mode in ["transcribe", "slack", "prompt", "polish"]:
        for line in (packets / f"{mode}.jsonl").read_text().splitlines():
            row = json.loads(line)
            (i, o, p), critique = critiques.pop(row["id"])
            reviews.append(
                {
                    "id": row["id"],
                    "row_sha256": row["row_sha256"],
                    "input_verdict": i,
                    "output_verdict": o,
                    "pair_verdict": p,
                    "verdict": "fail" if "fail" in (i, o, p) else "pass",
                    "critique": critique,
                }
            )
    assert not critiques and len(reviews) == report["coverage_count"] == 20
    report["reviews"] = reviews
    report["recovery"] = {
        "reason": "Reviewer constructed all 20 reviews but omitted the array from json.dump.",
        "source_command": "round3-train-dev-command.txt",
        "method": "AST literal parsing of the reviewer's C mapping and exact packet hashes.",
        "new_model_judgments": False,
    }
    destination = folder / "round3-train-dev-recovered.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"recovered_reviews": len(reviews), "report": str(destination)}))


if __name__ == "__main__":
    main()
