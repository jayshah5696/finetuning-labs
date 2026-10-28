"""Exercise the actual notebook cells with public Hub data and a tiny random CPU LFM.

The auth/install cells are skipped. W&B uses offline mode. The 230M weight loader
and final Hub model upload are replaced explicitly; adapter training/save/reload is real.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_DIR"] = str(args.output.resolve())
    import torch
    import wandb
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM, Lfm2Config, set_seed

    notebook_path = ROOT / "notebooks/rewriter_train_l4.ipynb"
    notebooks = sorted(
        p for directory in (ROOT / "notebooks", ROOT / "data") for p in directory.rglob("*.ipynb")
    )
    assert notebooks == [notebook_path], (
        "Keep one canonical notebook; retrieve old copies from Git history"
    )
    notebook = json.loads(notebook_path.read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
    for source in cells.values():
        ast.parse(source)
    helpers_path = ROOT / "src/finetuning_labs/rewriter/lora_training.py"
    assert ast.dump(ast.parse(cells["helpers"])) == ast.dump(ast.parse(helpers_path.read_text()))
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            assert c["execution_count"] is None and c["outputs"] == []
    assert "hf_hub_download" not in "\n".join(cells.values())
    assert "manifest.json" not in "\n".join(cells.values())
    scope = {"torch": torch, "wandb": wandb}

    def execute(identity):
        exec(compile(cells[identity], "notebook:" + identity, "exec"), scope)  # noqa: S102 -- trusted Git notebook is the integration target

    execute("config")
    execute("helpers")
    execute("data")
    original_counts = {k: len(v) for k, v in scope["encoded"].items()}
    max_tokens = max(len(r["input_ids"]) for split in scope["encoded"].values() for r in split)
    config = scope["CONFIG"]
    config.update(
        output_dir=str((args.output / "checkpoints").resolve()),
        epochs=2,
        batch_size=1,
        gradient_accumulation=1,
        run_name="cpu-lora-notebook-check",
    )
    scope["encoded"] = {s: rows.select(range(2)) for s, rows in scope["encoded"].items()}
    tokenizer = scope["tokenizer"]
    base_config = AutoConfig.from_pretrained(
        config["model_id"], revision=config["model_revision"], token=False
    )
    model_config = Lfm2Config(
        vocab_size=base_config.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        full_attn_idxs=[1],
        block_auto_adjust_ff_dim=False,
        block_multiple_of=8,
        max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    set_seed(config["seed"])
    base = AutoModelForCausalLM.from_config(model_config, attn_implementation="eager")
    base.name_or_path = config["model_id"]
    original_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    actual_attach = scope["attach_lora"]
    before_adapter = {}

    def attach(model, cfg):
        result = actual_attach(model, cfg)
        before_adapter.update(
            {k: p.detach().clone() for k, p in result.named_parameters() if p.requires_grad}
        )
        assert before_adapter and all("lora_" in k for k in before_adapter)
        return result

    scope["attach_lora"] = attach
    actual_trainer = scope["make_lora_trainer"]
    scope["make_lora_trainer"] = lambda *a, **k: actual_trainer(*a, **k, cpu=True)
    with patch.object(AutoModelForCausalLM, "from_pretrained", return_value=base):
        execute("train")
    model = scope["model"]
    delta = max(
        (dict(model.named_parameters())[k].detach() - v).abs().max().item()
        for k, v in before_adapter.items()
    )
    assert delta > 0
    parameters = dict(base.named_parameters(remove_duplicate=False))
    for name, value in original_state.items():
        # Linear base weights move under .base_layer when LoRA is attached.
        actual = parameters.get(name)
        if actual is None:
            prefix, suffix = name.rsplit(".", 1)
            actual = parameters[prefix + ".base_layer." + suffix]
        assert not actual.requires_grad and torch.equal(value, actual.detach()), name
    execute("save")
    assert "validation/loss" in dict(scope["run"].summary)
    final = scope["final_dir"]
    fresh = AutoModelForCausalLM.from_config(model_config, attn_implementation="eager")
    fresh.load_state_dict(original_state)
    reloaded = PeftModel.from_pretrained(fresh, final)
    trained_state = get_peft_model_state_dict(model, save_embedding_layers=False)
    loaded_state = get_peft_model_state_dict(reloaded, save_embedding_layers=False)
    from safetensors import safe_open

    with safe_open(final / "adapter_model.safetensors", framework="pt") as weights:
        keys = weights.keys()
        assert set(keys) == set(trained_state)
        assert all("lora_" in key for key in keys)
    assert trained_state.keys() == loaded_state.keys()
    assert all(torch.equal(v, loaded_state[k]) for k, v in trained_state.items())
    uploaded = {}

    class Hub:
        def create_repo(self, **kwargs):
            assert kwargs["repo_type"] == "model"
            uploaded["repo"] = kwargs

        def upload_folder(self, **kwargs):
            uploaded["files"] = kwargs["allow_patterns"]
            assert kwargs["repo_type"] == "model"
            return SimpleNamespace(commit_url="local-test://no-model-upload-performed")

    actual_push = scope["push_adapter"]
    scope["push_adapter"] = lambda *a, **k: actual_push(*a, **k, api=Hub())
    execute("upload")
    report = {
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "helpers_sha256": hashlib.sha256(
            (ROOT / "src/finetuning_labs/rewriter/lora_training.py").read_bytes()
        ).hexdigest(),
        "actual_public_hub_data": True,
        "notebook_count": len(notebooks),
        "public_download_token": False,
        "dataset_revision": config["dataset_revision"],
        "encoded_split_counts": original_counts,
        "max_chat_tokens": max_tokens,
        "tiny_random_native_lfm": True,
        "student_230m_weights_loaded": False,
        "gpu_run": False,
        "optimizer_steps": scope["trainer"].state.global_step,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_parameter_delta": delta,
        "base_weights_unchanged": True,
        "saved_adapter_reloaded_exactly": True,
        "wandb_mode": "offline",
        "wandb_validation_loss_logged": True,
        "actual_hub_model_upload": False,
        "adapter_upload_files": uploaded["files"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
