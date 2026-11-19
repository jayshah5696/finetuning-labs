"""Historical full-training utilities. The supported Colab workflow uses lora_training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tarfile
from pathlib import Path

MODEL = "LiquidAI/LFM2.5-230M"
MODEL_REVISION = "40cb2ad3b3044d5a41eee083a6103c8b523afa45"
MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048
FROZEN_VALIDATION = {
    "data/validation.jsonl": "37457d267394e2ac9b6a74463e1772063de6df8dbf09b781035d45894067484b",
    "sft/validation.jsonl": "9f238e291987ad2566b78780d2054e66c3a896f0cc09bef5e5331e36038df84c",
}


def sha(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def stage_finished(output):
    marker = output / "completed.json"
    if not marker.exists():
        return False
    for name, expected in json.loads(marker.read_text()).items():
        path = output / name
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError("A completed stage artifact changed or is missing")
    return True


def finish_stage(output, paths):
    write_json(output / "completed.json", {str(p.relative_to(output)): file_sha(p) for p in paths})


def save_recovery(local, persistent, step, best_checkpoint, binding_sha):
    """Rotate locally; overwrite two fixed Drive archives without deleting large files."""
    persistent.mkdir(parents=True, exist_ok=True)
    latest = local / f"checkpoint-{step}"
    paths = list(latest.rglob("*"))
    if best_checkpoint and Path(best_checkpoint) != latest:
        paths += [
            p
            for p in Path(best_checkpoint).rglob("*")
            if p.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"}
        ]
    temporary = local / "recovery.tmp.tar"
    with tarfile.open(temporary, "w") as archive:
        for path in sorted(set(paths)):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(local)), recursive=False)
    slots = [(json.loads(p.read_text())["step"], p.stem) for p in persistent.glob("slot-*.json")]
    slot = "slot-0" if not slots else ("slot-1" if max(slots)[1] == "slot-0" else "slot-0")
    target = persistent / (slot + ".tar")
    with temporary.open("rb") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    write_json(
        persistent / (slot + ".json"),
        {
            "step": step,
            "archive": target.name,
            "sha256": file_sha(temporary),
            "binding_sha256": binding_sha,
        },
    )
    temporary.unlink()


def restore_recovery(persistent, local, binding_sha):
    candidates = []
    for marker in persistent.glob("slot-*.json"):
        record = json.loads(marker.read_text())
        if record["binding_sha256"] != binding_sha:
            raise ValueError("Recovery binding differs")
        archive = persistent / record["archive"]
        if archive.is_file() and file_sha(archive) == record["sha256"]:
            candidates.append((record["step"], archive))
        else:
            print(f"Ignoring incomplete recovery archive: {archive.name}", flush=True)
            # Keep the failed receipt as evidence, but never select its slot as
            # the known-good archive during the next alternating overwrite.
            marker.replace(marker.with_name(marker.name + ".invalid-" + file_sha(marker)[:12]))
    if not candidates:
        return None
    step, archive = max(candidates)
    local.mkdir(parents=True, exist_ok=True)
    if any(local.iterdir()):
        raise FileExistsError("Restore requires an empty local checkpoint directory")
    with tarfile.open(archive) as saved:
        for member in saved.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError("Unsafe checkpoint archive member")
        saved.extractall(local, filter="data")
    return local / f"checkpoint-{step}"


def recovery_callback(persistent, binding_sha):
    from transformers import TrainerCallback

    class Recovery(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            save_recovery(
                Path(args.output_dir),
                persistent,
                state.global_step,
                state.best_model_checkpoint,
                binding_sha,
            )

    return Recovery()


def greedy_config(tokenizer):
    from transformers import GenerationConfig

    config = GenerationConfig(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1.0,
        use_cache=True,
    )
    # Transformers 5 fills unset fields from model defaults. Materialize the
    # pinned library's defaults and replace the model config as well.
    config.update(**config._get_default_generation_params(), defaults_only=True)
    config.validate(strict=True)
    return config


def read_saved_outputs(path):
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        boundary = data.rfind(b"\n") + 1
        partial = data[boundary:]
        quarantine = path.with_name(
            "outputs.partial-" + hashlib.sha256(partial).hexdigest() + ".bin"
        )
        quarantine.write_bytes(partial)
        with path.open("r+b") as stream:
            stream.truncate(boundary)
        data = data[:boundary]
    return [json.loads(line) for line in data.splitlines()]


def inference_messages(guide):
    return [
        {"role": "system", "content": guide["system"]},
        {"role": "user", "content": guide["user"]},
    ]


def encode_example(guide, tokenizer, *, max_length=MAX_LENGTH, max_new_tokens=MAX_NEW_TOKENS):
    if set(guide) != {"system", "user", "assistant"}:
        raise ValueError("SFT rows must have exactly system, user and assistant")
    messages = inference_messages(guide)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": guide["assistant"]}],
        tokenize=True,
        return_dict=False,
    )
    if len(full) > max_length or len(prompt) + max_new_tokens > max_length:
        raise ValueError("context overflow; refusing truncation")
    if full[: len(prompt)] != prompt:
        raise ValueError("assistant prefix differs from inference prefix")
    rest = full[len(prompt) :]
    if tokenizer.eos_token_id not in rest:
        raise ValueError("assistant EOS missing")
    end = len(prompt) + rest.index(tokenizer.eos_token_id) + 1
    if end <= len(prompt) + 1 or end - len(prompt) > max_new_tokens:
        raise ValueError("empty or oversized assistant span")
    if end < len(full) and tokenizer.decode(full[end:]).strip():
        raise ValueError("unexpected content after assistant EOS")
    labels = [-100] * len(prompt) + full[len(prompt) : end] + [-100] * (len(full) - end)
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def pad_batch(features, pad_id):
    if not features:
        raise ValueError("features cannot be empty")
    width = max(len(x["input_ids"]) for x in features)
    return {
        key: [x[key] + [padding] * (width - len(x[key])) for x in features]
        for key, padding in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]
    }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in pad_batch(features, self.pad_id).items()
        }


def bind_run(output, config, *, resume):
    output = Path(output)
    binding = output / "run.json"
    if binding.exists():
        if json.loads(binding.read_text()) != config:
            raise ValueError("resume configuration or dataset differs")
        if not resume:
            raise FileExistsError("Run exists; use --resume or a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output contains unbound files")
        if resume:
            raise ValueError("Cannot resume without the original run binding")
        output.mkdir(parents=True, exist_ok=True)
        write_json(binding, config)


def make_trainer(
    model, tokenizer, train, dev, output, *, proof=False, cpu=False, max_steps=None, callbacks=None
):
    from transformers import Trainer, TrainingArguments

    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=3,
        max_steps=(80 if proof else -1) if max_steps is None else max_steps,
        learning_rate=1e-4 if proof else 5e-5,
        weight_decay=0.01,
        warmup_steps=0.05,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if proof else 16,
        bf16=not cpu,
        fp16=False,
        use_cpu=cpu,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no" if proof else "steps",
        save_steps=20,
        save_total_limit=1,
        load_best_model_at_end=not proof,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1 if proof else 10,
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    model.config.use_cache = False
    model.generation_config = greedy_config(tokenizer)
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=Collator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=callbacks,
    )


def proof_indices(metadata, guides):
    selected = []
    for mode in ["/slack", "/transcribe", "/prompt", "/polish"]:
        choices = [i for i, row in enumerate(guides) if row["user"].startswith(f"MODE: {mode}\n")]
        for identity in [True, False]:
            short = [
                i
                for i in choices
                if metadata[i]["length"] == "short"
                and bool(metadata[i].get("identity")) == identity
            ]
            if not short:
                raise ValueError(f"Proof needs short identity={identity} in {mode}")
            selected.append(short[0])
        long = [i for i in choices if metadata[i]["length"] == "long"]
        long.sort(key=lambda i: not bool(metadata[i].get("identity")))
        if len(long) < 2:
            raise ValueError("Proof needs two long examples in each mode")
        selected.extend(long[:2])
    if len(set(selected)) != 16:
        raise ValueError("Proof selection must have 16 distinct training examples")
    return selected


def environment():
    import torch

    return {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def require_gpu():
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select an L4 GPU runtime with BF16 support in Colab")


def generate(model, tokenizer, guides, metadata, output, *, resume):
    import torch

    destination = output / "outputs.jsonl"
    completed = {}
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        for record in read_saved_outputs(destination):
            if record["case_id"] in completed:
                raise ValueError("Duplicate saved generation")
            if record.get("result_sha256") != sha(
                {k: v for k, v in record.items() if k != "result_sha256"}
            ):
                raise ValueError("Saved generation was edited")
            completed[record["case_id"]] = record
    if set(completed) - {m["id"] for m in metadata}:
        raise ValueError("Foreign case in saved generations")
    model.eval()
    model.config.use_cache = True
    decoding = greedy_config(tokenizer)
    model.generation_config = decoding
    for guide, meta in zip(guides, metadata, strict=True):
        request_sha = sha(inference_messages(guide))
        if meta["id"] in completed:
            if completed[meta["id"]]["request_sha256"] != request_sha:
                raise ValueError("Saved generation belongs to a different input")
            continue
        inputs = tokenizer.apply_chat_template(
            inference_messages(guide),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if inputs["input_ids"].shape[1] + MAX_NEW_TOKENS > MAX_LENGTH:
            raise ValueError("Generation context overflow")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            full = model.generate(
                **inputs,
                generation_config=decoding,
            )
        new = full[0, inputs["input_ids"].shape[1] :].tolist()
        record = {
            "case_id": meta["id"],
            "mode": meta.get("mode", guide["user"].splitlines()[0][6:]),
            "request_sha256": request_sha,
            "row_sha256": meta["row_sha256"],
            "text": tokenizer.decode(new, skip_special_tokens=True),
            "status": "completed" if new and new[-1] == tokenizer.eos_token_id else "truncated",
            "generated_tokens": len(new),
            "generation_config": decoding.to_dict(),
        }
        record["result_sha256"] = sha(record)
        with destination.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"{meta['id']}: {record['status']}", flush=True)


def main():
    raise SystemExit(
        "Retired full-training runner. Use notebooks/rewriter_train_l4.ipynb for public Parquet data and PEFT LoRA."
    )


if __name__ == "__main__":
    main()
