"""Standalone Colab L4 training entry point for the frozen reviewed 1,000-pair release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

MODEL = "LiquidAI/LFM2.5-230M"
MODEL_REVISION = "40cb2ad3b3044d5a41eee083a6103c8b523afa45"
REPO = "jayshah5696/finetuning-labs-rewriter-training"
PREFIX = "reviewed-v3-1000"
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


def load_release(revision, token):
    from huggingface_hub import hf_hub_download

    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An immutable 40-character dataset revision is required")

    def download(name):
        path = hf_hub_download(
            REPO, f"{PREFIX}/{name}", repo_type="dataset", revision=revision, token=token
        )
        return Path(path).read_bytes()

    raw_manifest = download("manifest.json")
    manifest = json.loads(raw_manifest)

    def checked(name):
        data = download(name)
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Dataset file hash differs: {name}")
        if (
            name in FROZEN_VALIDATION
            and hashlib.sha256(data).hexdigest() != FROZEN_VALIDATION[name]
        ):
            raise ValueError("Frozen v2 validation bytes changed")
        return data

    quality = json.loads(checked("quality.json"))
    if quality["splits"] != {"train": 1000, "dev": 100, "validation": 84}:
        raise ValueError("Expected 1,000 train, 100 dev and 84 validation pairs")
    support = json.loads(checked("supporting.json"))
    guides, metadata, groups, identities = {}, {}, {}, set()
    for split in ["train", "dev", "validation"]:
        rows = [json.loads(x) for x in checked(f"data/{split}.jsonl").splitlines()]
        guides[split] = [json.loads(x) for x in checked(f"sft/{split}.jsonl").splitlines()]
        metadata[split] = [m for m in support if m["split"] == split]
        if not (
            len(rows) == len(guides[split]) == len(metadata[split]) == quality["splits"][split]
        ):
            raise ValueError("Split row counts differ")
        groups[split] = set()
        for row, guide, meta in zip(rows, guides[split], metadata[split], strict=True):
            if set(row) != {"input", "output", "mode", "checks"} or sha(row) != meta["row_sha256"]:
                raise ValueError("Row schema or identity differs")
            if meta["id"] in identities:
                raise ValueError("Duplicate source identity")
            identities.add(meta["id"])
            groups[split].add(meta["source_group"])
            review = meta["opus_review"]
            if (
                (review["id"], review["row_sha256"]) != (meta["id"], sha(row))
                or any(
                    review[k] != "pass"
                    for k in ["input_verdict", "output_verdict", "pair_verdict", "verdict"]
                )
                or meta["mechanical_issues"]
            ):
                raise ValueError("Unverified row in release")
            if guide["assistant"] != row["output"] or set(guide) != {"system", "user", "assistant"}:
                raise ValueError("Supervised target differs from verified data")
            header = [f"MODE: {row['mode']}", f"LANGUAGE: {meta['language']}"]
            if meta["context"].get("profile"):
                header.append("PROFILE: " + meta["context"]["profile"])
            if meta["context"].get("dictionary"):
                header.append("DICTIONARY: " + ", ".join(meta["context"]["dictionary"]))
            if guide["user"] != "\n".join([*header, "INPUT:", row["input"]]):
                raise ValueError("Input or required context differs")
    if len(identities) != len(support):
        raise ValueError("Foreign metadata in release")
    for a, b in [("train", "dev"), ("train", "validation"), ("dev", "validation")]:
        if not groups[a].isdisjoint(groups[b]):
            raise ValueError("Source group crosses splits")
    return guides, metadata, hashlib.sha256(raw_manifest).hexdigest()


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
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "baseline", "proof", "train", "tuned"])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("/content/rewriter-checkpoints"))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    guides, metadata, manifest_sha = load_release(args.revision, token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = {s: [encode_example(g, tokenizer) for g in rows] for s, rows in guides.items()}
    binding = {
        "stage": args.stage,
        "dataset_revision": args.revision,
        "manifest_sha256": manifest_sha,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": 42,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generation_config": greedy_config(tokenizer).to_dict(),
        "packages": {
            k: importlib.metadata.version(k)
            for k in ["transformers", "accelerate", "tokenizers", "safetensors"]
        },
    }
    if args.stage in {"proof", "train"}:
        binding["local_work_dir"] = str(args.work_dir.resolve())
    if args.stage == "check":
        bind_run(args.output, binding, resume=args.resume)
        report = dict(
            binding,
            splits={k: len(v) for k, v in encoded.items()},
            max_chat_tokens=max(len(r["input_ids"]) for v in encoded.values() for r in v),
            assistant_only_labels=True,
            proof_ids=[
                metadata["train"][i]["id"]
                for i in proof_indices(metadata["train"], guides["train"])
            ],
            student_training=False,
        )
        (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    import torch

    if args.stage == "tuned":
        if not args.checkpoint or not (args.checkpoint / "config.json").is_file():
            raise ValueError("Tuned generation requires the saved selected-model directory")
        weights = list(args.checkpoint.glob("*.safetensors"))
        if not weights:
            raise ValueError("Checkpoint has no safetensors weights")
        if not (args.checkpoint / "generation_config.json").is_file():
            raise ValueError("Checkpoint has no generation config")
        binding["checkpoint_hashes"] = {
            p.name: file_sha(p) for p in args.checkpoint.iterdir() if p.is_file()
        }
    elif args.checkpoint:
        raise ValueError(
            "Only tuned generation accepts --checkpoint; training starts from the pinned base"
        )
    bind_run(args.output, binding, resume=args.resume)
    if stage_finished(args.output):
        print(f"{args.stage}: already completed; verified saved artifacts", flush=True)
        return
    require_gpu()
    set_seed(42)
    sessions = args.output / "sessions"
    sessions.mkdir(exist_ok=True)
    write_json(sessions / f"{time.time_ns()}.json", environment())
    training = args.stage in {"proof", "train"}
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint) if args.stage == "tuned" else MODEL,
        **({} if args.stage == "tuned" else {"revision": MODEL_REVISION, "token": token}),
        dtype=torch.float32 if training else torch.bfloat16,
        attn_implementation="eager",
    )
    if training:
        train, dev = encoded["train"], encoded["dev"]
        if args.stage == "proof":
            indices = proof_indices(metadata["train"], guides["train"])
            train = [train[i] for i in indices]
            dev = train
        proof = args.stage == "proof"
        local = args.work_dir / sha(binding)
        local.parent.mkdir(parents=True, exist_ok=True)
        # Only this script's hash-bound local scratch directory is cleared.
        if local.exists():
            shutil.rmtree(local)
        local.mkdir()
        for path, required_gib in [(local, 12), (args.output, 10 if not proof else 1)]:
            reusable = (
                sum(p.stat().st_size for p in (args.output / "recovery").glob("slot-*.tar"))
                if path == args.output
                else 0
            )
            if shutil.disk_usage(path).free + reusable < required_gib * 1024**3:
                raise RuntimeError(f"Need {required_gib} GiB free at {path}")
        recovery = args.output / "recovery"
        checkpoint = (
            restore_recovery(recovery, local, sha(binding)) if args.resume and not proof else None
        )
        print(
            f"Training {'resumes from ' + str(checkpoint) if checkpoint else 'starts from pinned base (step 0)'}",
            flush=True,
        )
        trainer = make_trainer(
            model,
            tokenizer,
            train,
            dev,
            local,
            proof=proof,
            callbacks=[] if proof else [recovery_callback(recovery, sha(binding))],
        )
        result = trainer.train(resume_from_checkpoint=checkpoint)
        if not proof:
            trainer.save_model(str(args.output / "selected-model"))
            tokenizer.save_pretrained(args.output / "selected-model")
        trainer.state.save_to_json(str(args.output / "trainer_state.json"))
        report = {
            "metrics": result.metrics,
            "checkpoint_selection": "training memorization only"
            if args.stage == "proof"
            else "dev loss only",
            "validation_used_for_training": False,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "environment": environment(),
            "resumed_from": str(checkpoint) if checkpoint else None,
            "proof_checkpoints_saved": False,
        }
        write_json(args.output / "training-result.json", report)
        artifacts = [args.output / "training-result.json", args.output / "trainer_state.json"]
        if not proof:
            artifacts += [p for p in (args.output / "selected-model").iterdir() if p.is_file()]
        finish_stage(args.output, artifacts)
    else:
        model.to("cuda")
        generate(
            model,
            tokenizer,
            guides["validation"],
            metadata["validation"],
            args.output,
            resume=args.resume,
        )
        finish_stage(args.output, [args.output / "outputs.jsonl"])


if __name__ == "__main__":
    main()
