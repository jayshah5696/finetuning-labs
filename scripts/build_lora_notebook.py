"""Build the self-contained Colab notebook from the tested LoRA helpers in Git."""

import argparse
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, identity, source, *, hidden=False):
    value = {
        "cell_type": kind,
        "id": identity,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
        if hidden:
            value["metadata"]["cellView"] = "form"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args()
    receipt = json.loads((ROOT / "data/rewriter/releases/public-parquet.json").read_text())
    published = next(
        r
        for r in receipt["repositories"]
        if r["repo_id"] == "jayshah5696/finetuning-labs-rewriter-training"
    )
    if published["revision"] != args.dataset_revision:
        raise ValueError("Build the notebook from the current published dataset receipt")
    config = {
        "model_id": "LiquidAI/LFM2.5-230M",
        "model_revision": "40cb2ad3b3044d5a41eee083a6103c8b523afa45",
        "dataset_id": "jayshah5696/finetuning-labs-rewriter-training",
        "dataset_revision": args.dataset_revision,
        "dataset_config": "sft",
        "expected_split_counts": published["splits"],
        "hub_model_id": "jayshah5696/lfm2.5-230m-rewriter-lora",
        "hub_private": False,
        "wandb_project": "finetuning-labs",
        "run_name": "lfm2.5-230m-rewriter-lora",
        "output_dir": "/content/rewriter-lora",
        "epochs": 3,
        "learning_rate": 0.0002,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_length": 4096,
        "seed": 42,
    }
    config_source = (
        "# Change model, LoRA and upload settings here.\nCONFIG = {\n"
        + "".join(f"    {key!r}: {value!r},\n" for key, value in config.items())
        + "}\n"
    )
    ast.parse(config_source)
    cells = [
        cell(
            "markdown",
            "intro",
            """
# Train and upload a rewriter LoRA

Notebook version: `public-parquet-lora-current`. Training code is included below and generated from Git.

Choose an L4 GPU in Colab. Add these two Colab Secrets and enable notebook access:

- `HF_TOKEN`: a Hugging Face token with **write access** to the model repository in CONFIG.
- `WANDB_API_KEY`: your Weights & Biases API key.

Run the cells in order. This notebook downloads the public train/dev/validation data, trains a
LoRA adapter on the configured model, logs loss to W&B and uploads the final adapter to the Hub.
Change `hub_model_id` if you use a different Hugging Face account. Code stays in Git.
""",
        ),
        cell(
            "code",
            "install",
            """
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.21.0", "peft==0.21.2", "datasets==5.1.0",
    "accelerate==1.15.0", "wandb==0.31.0", "huggingface_hub==1.33.0",
    "tokenizers==0.23.3", "safetensors==0.8.0", "jinja2==3.1.6"])
""",
        ),
        cell("code", "config", config_source),
        cell(
            "code",
            "auth",
            """
import os
import torch
import wandb
from google.colab import userdata
from huggingface_hub import login

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("Choose an L4 GPU runtime before training.")

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = CONFIG["wandb_project"]
os.environ["WANDB_LOG_MODEL"] = "false"
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
wandb.login(key=os.environ["WANDB_API_KEY"])
print("GPU:", torch.cuda.get_device_name(0))
""",
        ),
        cell(
            "code",
            "helpers",
            "#@title Training helpers\n"
            + (ROOT / "src/finetuning_labs/rewriter/lora_training.py").read_text(),
            hidden=True,
        ),
        cell(
            "code",
            "data",
            """
from datasets import load_dataset
from transformers import AutoTokenizer

raw = load_dataset(CONFIG["dataset_id"], CONFIG["dataset_config"],
                   revision=CONFIG["dataset_revision"], token=False)
print({split: len(rows) for split, rows in raw.items()})
assert {split: len(rows) for split, rows in raw.items()} == CONFIG["expected_split_counts"]
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], revision=CONFIG["model_revision"],
                                         token=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
encoded = raw.map(lambda row: encode_sft(row, tokenizer, CONFIG["max_length"]),
                  remove_columns=raw["train"].column_names, desc="Tokenize assistant-only labels")
""",
        ),
        cell(
            "markdown",
            "training-note",
            """
Training uses `train`. Checkpoints are selected by loss on `dev`. The frozen `validation` split
is evaluated once after training. Loss is logged to W&B; it is not a rewrite pass-rate metric.
Only LoRA parameters are updated. Checkpoints stay on local Colab storage.
""",
        ),
        cell(
            "code",
            "train",
            """
from transformers import AutoModelForCausalLM, set_seed

set_seed(CONFIG["seed"])
model = AutoModelForCausalLM.from_pretrained(CONFIG["model_id"],
    revision=CONFIG["model_revision"], token=False,
    dtype=torch.bfloat16, attn_implementation="eager")
model = attach_lora(model, CONFIG)
run = wandb.init(project=CONFIG["wandb_project"], name=CONFIG["run_name"], config=CONFIG)
trainer = make_lora_trainer(model, tokenizer, encoded["train"], encoded["dev"], CONFIG)
train_result = trainer.train()
""",
        ),
        cell(
            "code",
            "save",
            """
metrics = trainer.evaluate(encoded["validation"], metric_key_prefix="validation")
wandb.log({"validation/loss": metrics["validation_loss"]})
final_dir = Path(CONFIG["output_dir"]) / "final_adapter"
model.save_pretrained(final_dir, safe_serialization=True, save_embedding_layers=False)
tokenizer.save_pretrained(final_dir)
card = f'''---
library_name: peft
base_model: {CONFIG["model_id"]}
datasets:
- {CONFIG["dataset_id"]}
tags: [lora, text-rewriting]
pipeline_tag: text-generation
---
# Rewriter LoRA

LoRA adapter for `{CONFIG["model_id"]}` at revision `{CONFIG["model_revision"]}`.
Training data: `{CONFIG["dataset_id"]}` / `sft` at `{CONFIG["dataset_revision"]}`.
Train rows: {len(encoded["train"])}. Dev rows: {len(encoded["dev"])}.
Rank: {CONFIG["lora_r"]}. Alpha: {CONFIG["lora_alpha"]}.
The checkpoint was selected by dev loss. Final validation loss: {metrics["validation_loss"]:.6f}.
This loss does not establish semantic rewrite quality. The original base model is required.
The base model's license and dataset source notices still apply.
'''
(final_dir / "README.md").write_text(card)
print("Final adapter:", final_dir)
print("Files:", adapter_files(final_dir))
""",
        ),
        cell(
            "code",
            "upload",
            """
commit = push_adapter(final_dir, CONFIG["hub_model_id"], private=CONFIG["hub_private"])
print("Uploaded:", commit.commit_url)
wandb.finish()
""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks/rewriter_train_l4.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    subprocess.run(
        ["uv", "run", "--locked", "ruff", "check", "--fix", str(output)], cwd=ROOT, check=True
    )
    subprocess.run(["uv", "run", "--locked", "ruff", "format", str(output)], cwd=ROOT, check=True)
    print(output)


if __name__ == "__main__":
    main()
