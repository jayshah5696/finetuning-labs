"""Run the same Liquid baseline on an L4 and collect durable artifacts."""

import json
import tomllib
from pathlib import Path

import modal

from finetuning_labs.rewriter.evaluation import (
    code_provenance,
    generate_cases,
    liquid_backend,
)

PROJECT = Path(__file__).resolve().parents[1]
app = modal.App("finetuning-labs")
volume = modal.Volume.from_name("finetuning-labs-evals", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "transformers==5.21.0", "jinja2==3.1.6")
    .env({"HF_HOME": "/runs/hf-cache"})
    .add_local_python_source("finetuning_labs")
    .add_local_dir(PROJECT / "data/rewriter/benchmark-draft", "/bundle")
)


@app.function(image=image, gpu="L4", cpu=4, memory=12288, timeout=1800, volumes={"/runs": volume})
def execute(run_id: str, config: dict):
    from finetuning_labs.rewriter.benchmark import load_bundle

    bundle = load_bundle(Path("/bundle"))
    destination = Path("/runs") / run_id / "liquid"
    try:
        result = generate_cases(bundle, destination, config, liquid_backend(config))
        return {"case_count": len(result), "output": str(destination)}
    finally:
        volume.commit()


@app.local_entrypoint()
def submit(run_id: str = "baseline-20261027"):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID with digits, hyphens, or underscores")
    destination = PROJECT / "data/rewriter/evals" / run_id
    receipt_path = destination / "modal.json"
    if receipt_path.exists():
        raise FileExistsError("A run receipt already exists; collect it instead of resubmitting")
    settings = tomllib.loads((PROJECT / "configs/rewriter.toml").read_text())
    config = settings["liquid"] | code_provenance() | {"execution": "modal-L4"}
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "submitting",
        "app_id": app.app_id,
        "app_name": app.name,
        "function": "execute",
        "run_id": run_id,
        "volume": "finetuning-labs-evals",
        "remote_root": f"{run_id}/liquid",
        "config": config,
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    call = execute.spawn(run_id, config)
    receipt.update(status="submitted", function_call_id=call.object_id)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


@app.local_entrypoint()
def collect(run_id: str = "baseline-20261027"):
    destination = PROJECT / "data/rewriter/evals" / run_id
    receipt_path = destination / "modal.json"
    receipt = json.loads(receipt_path.read_text())
    call = modal.FunctionCall.from_id(receipt["function_call_id"])
    result = call.get()
    root = receipt["remote_root"]
    for entry in volume.iterdir(root, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path).relative_to(root)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe result path")
        target = destination / "liquid" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for part in volume.read_file(entry.path):
                stream.write(part)
    receipt.update(status="collected", result=result)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
