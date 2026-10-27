"""Run the actual notebook data cells locally against the private Hub release.

Only Colab Secrets is adapted to cached Hugging Face authentication. Skip the pip-install
cell because uv supplies the locked environment. This does not establish Colab GPU execution.
"""

import hashlib
import json
import os
import sys
import types
from pathlib import Path

from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook_path = ROOT / "notebooks/rewriter_reviewed_data_check.ipynb"
    notebook = json.loads(notebook_path.read_text())
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"])
    token = get_token()
    if not token:
        raise ValueError("Cached private Hub authentication is required")
    adapter = types.ModuleType("google.colab")
    adapter.userdata = types.SimpleNamespace(get=lambda name: token if name == "HF_TOKEN" else None)
    sys.modules["google.colab"] = adapter
    output = ROOT / "artifacts/colab-reviewed-hub-check"
    output.mkdir(exist_ok=True)
    os.chdir(output)
    namespace = {}
    executed = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compiled = compile(source, f"{notebook_path.name}:cell-{index}", "exec")
            if index != 1:
                exec(compiled, namespace)  # noqa: S102 - Run the committed notebook cells.
                executed += 1
    assert executed == 3
    receipt = json.loads(Path("rewriter-data-readiness.json").read_text())
    receipt.update(
        execution_environment="local locked uv environment; cached Hub auth replaces Colab Secrets",
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        notebook_data_cells_executed=executed,
        actual_colab_execution=False,
        gpu_training=False,
    )
    destination = ROOT / "data/rewriter/releases/colab-reviewed-data-check-local.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verified_against_hub": receipt["revision"],
                "executed_data_cells": executed,
                "actual_colab_execution": False,
            }
        )
    )


if __name__ == "__main__":
    main()
