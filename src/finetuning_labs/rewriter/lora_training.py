"""Small helpers embedded in the self-contained Colab LoRA notebook."""

import json
from pathlib import Path


def encode_sft(row, tokenizer, max_length=4096):
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["assistant"]}],
        tokenize=True,
        return_dict=False,
    )
    if len(full) > max_length:
        raise ValueError(
            "Example exceeds max_length; increase the config rather than truncate data"
        )
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template changed the assistant prefix")
    tail = full[len(prompt) :]
    if tokenizer.eos_token_id not in tail or tail.index(tokenizer.eos_token_id) == 0:
        raise ValueError("Assistant text and its EOS are required")
    end = len(prompt) + tail.index(tokenizer.eos_token_id) + 1
    if end < len(full) and tokenizer.decode(full[end:]).strip():
        raise ValueError("Unexpected text after assistant EOS")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt) + full[len(prompt) : end] + [-100] * (len(full) - end),
    }


class SFTCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch

        size = max(len(row["input_ids"]) for row in rows)
        return {
            key: torch.tensor([row[key] + [value] * (size - len(row[key])) for row in rows])
            for key, value in [
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }


def attach_lora(model, config):
    from peft import LoraConfig, TaskType, get_peft_model

    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules="all-linear",
        bias="none",
        revision=config["model_revision"],
    )
    model = get_peft_model(model, adapter)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def make_lora_trainer(model, tokenizer, train, dev, config, *, cpu=False):
    from transformers import Trainer, TrainingArguments

    arguments = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"],
        warmup_steps=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=not cpu,
        fp16=False,
        use_cpu=cpu,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="wandb",
        run_name=config["run_name"],
        seed=config["seed"],
        data_seed=config["seed"],
        remove_unused_columns=False,
    )
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )


def adapter_files(directory):
    directory = Path(directory)
    required = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
    allowed = required | {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
    }
    present = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if not required.issubset(present) or not present.issubset(allowed):
        raise ValueError(
            "Final adapter folder must contain adapter weights, config and tokenizer only"
        )
    if json.loads((directory / "adapter_config.json").read_text()).get("peft_type") != "LORA":
        raise ValueError("Expected a LoRA adapter config")
    return sorted(present)


def push_adapter(directory, repo_id, *, private=False, api=None):
    from huggingface_hub import HfApi

    files = adapter_files(directory)
    api = api or HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(directory),
        allow_patterns=files,
        commit_message="Upload trained LoRA adapter",
    )
