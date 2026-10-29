"""Run the same Liquid baseline on an L4 and collect durable artifacts."""

import json
import tomllib
from pathlib import Path

import modal

from finetuning_labs.rewriter.evaluation import (
    bundle_identity,
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
    .uv_pip_install("peft==0.21.2", "accelerate==1.15.0")
    .env({"HF_HOME": "/runs/hf-cache"})
    .add_local_python_source("finetuning_labs")
    .add_local_dir(PROJECT / "data/rewriter/benchmark-draft", "/bundle")
    .add_local_dir(PROJECT / "data/rewriter/reviewed-v2/validation-bundle", "/validation-v2")
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


@app.function(image=image, gpu="L4", cpu=2, memory=6144, timeout=7200, volumes={"/runs": volume})
def execute_lora(run_id: str, config: dict, adapter: dict):
    import gc

    import torch

    from finetuning_labs.rewriter.benchmark import load_bundle

    bundle = load_bundle(Path("/validation-v2"))
    if len(bundle.cases) != 84 or bundle_identity(bundle) != config["source_bundle_sha256"]:
        raise ValueError("Remote validation bundle differs from the frozen local bundle")
    result = {}
    for arm, settings in [("stock", config), ("lora", config | adapter)]:
        destination = Path("/runs") / run_id / arm
        print(f"Starting {arm}: {len(bundle.cases)} frozen validation cases", flush=True)
        try:
            backend = liquid_backend(settings)
            rows = generate_cases(bundle, destination, settings, backend)
            result[arm] = {"case_count": len(rows), "output": str(destination)}
            del backend
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            volume.commit()
    return result


@app.local_entrypoint()
def submit_lora(
    run_id: str = "lora-validation-20261028",
    gpu: str = "L4",
    adapter_revision: str = "ba71856780e4d2398218fe76756b3390556437ea",
    dataset_revision: str = "139bb98b0500fa12ad43dc6c8c5cd4d17645eb86",
):
    import hashlib

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from finetuning_labs.rewriter.benchmark import COMMON_SYSTEM, MODES, load_bundle

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID with digits, hyphens, or underscores")
    devices = {"L4": "L4", "A10G": "A10"}
    if gpu not in devices:
        raise ValueError("Use L4 or A10G for matched BF16 evaluation")
    for revision in [adapter_revision, dataset_revision]:
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Pin adapter and dataset to immutable 40-character Hub commits")
    destination = PROJECT / "data/rewriter/evals" / run_id
    receipt_path = destination / "modal.json"
    if receipt_path.exists():
        raise FileExistsError("A run receipt exists; collect it instead of resubmitting")
    data = Path(
        hf_hub_download(
            "jayshah5696/finetuning-labs-rewriter-training",
            "sft/validation.parquet",
            repo_type="dataset",
            revision=dataset_revision,
            token=False,
        )
    )
    bundle = load_bundle(PROJECT / "data/rewriter/reviewed-v2/validation-bundle")
    expected = [
        {"system": COMMON_SYSTEM + MODES[c.mode], "user": c.user_message(), "assistant": c.output}
        for c in bundle.cases
    ]
    if pq.read_table(data).to_pylist() != expected:
        raise ValueError("Frozen validation prompts or references differ from public Hub data")
    settings = tomllib.loads((PROJECT / "configs/rewriter.toml").read_text())
    config = (
        settings["liquid"]
        | code_provenance()
        | {
            "execution": f"modal-{gpu}",
            "expected_gpu": devices[gpu],
            "source_bundle_sha256": bundle_identity(bundle),
            "dataset_revision": dataset_revision,
            "validation_parquet_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        }
    )
    adapter = {
        "adapter_model": "jayshah5696/lfm2.5-230m-rewriter-lora",
        "adapter_revision": adapter_revision,
    }
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "submitting",
        "app_id": app.app_id,
        "app_name": app.name,
        "function": "execute_lora",
        "run_id": run_id,
        "volume": "finetuning-labs-evals",
        "remote_root": run_id,
        "config": config,
        "adapter": adapter,
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    call = execute_lora.with_options(gpu=gpu).spawn(run_id, config, adapter)
    receipt.update(status="submitted", function_call_id=call.object_id)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


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
        target = (
            destination / relative
            if receipt["function"] == "execute_lora"
            else destination / "liquid" / relative
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(volume.read_file(entry.path))
        if target.exists() and target.read_bytes() == data:
            continue
        temporary = target.with_suffix(target.suffix + ".collecting")
        temporary.write_bytes(data)
        temporary.replace(target)
    receipt.update(status="collected", result=result)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
