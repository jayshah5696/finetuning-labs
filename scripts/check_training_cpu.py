"""Exercise the real Trainer, collator and checkpoint with a tiny RANDOM Liquid model.

No student weights are downloaded. This is an implementation check, not a fine-tune result.
Run in a uv environment with torch, transformers and accelerate installed.
"""

import importlib.util
import json
import math
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Lfm2Config,
    TrainerCallback,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "src/finetuning_labs/rewriter/colab_training.py"
    spec = importlib.util.spec_from_file_location("colab_training", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(module.MODEL, revision=module.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    guides = [
        json.loads(x)
        for x in (ROOT / "data/rewriter/reviewed-v2/sft/train.jsonl").read_text().splitlines()
    ][:2]
    rows = [module.encode_example(x, tokenizer) for x in guides]
    config = Lfm2Config(
        vocab_size=len(tokenizer),
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
    set_seed(42)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    before = next(model.parameters()).detach().clone()
    output = ROOT / "artifacts/training-cpu-integration-v3"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    local = output / "local"
    recovery = output / "recovery"
    pinned_generation = hf_hub_download(
        module.MODEL, "generation_config.json", revision=module.MODEL_REVISION
    )
    model.generation_config = GenerationConfig.from_pretrained(
        module.MODEL, revision=module.MODEL_REVISION
    )

    class Interrupt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True

    def build(model, callbacks):
        trainer = module.make_trainer(
            model, tokenizer, rows, rows, local, cpu=True, max_steps=2, callbacks=callbacks
        )
        trainer.args.save_steps = trainer.args.eval_steps = 1
        return trainer

    trainer = build(model, [module.recovery_callback(recovery, "cpu-integration"), Interrupt()])
    trainer.train()
    assert trainer.state.global_step == 1
    shutil.rmtree(local)
    checkpoint = module.restore_recovery(recovery, local, "cpu-integration")
    reinitialized = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    trainer = build(reinitialized, [module.recovery_callback(recovery, "cpu-integration")])
    model = reinitialized
    result = trainer.train(resume_from_checkpoint=checkpoint)
    assert math.isfinite(result.training_loss)
    parameter_delta = (next(model.parameters()).detach() - before).abs().max().item()
    assert parameter_delta > 0
    trainer.save_model(output / "selected-model")
    reloaded = AutoModelForCausalLM.from_pretrained(output / "selected-model")
    batch = module.Collator(tokenizer.pad_token_id)(rows)
    loss = reloaded(**batch).loss.item()
    assert math.isfinite(loss)
    report = {
        "random_tiny_model": True,
        "student_weights_loaded": False,
        "optimizer_steps": trainer.state.global_step,
        "maximum_parameter_change": parameter_delta,
        "training_loss": result.training_loss,
        "reloaded_checkpoint_loss": loss,
        "assistant_mask_and_collator_used": True,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "environment": module.environment(),
        "actual_colab_execution": False,
        "interrupted_after_step": 1,
        "resumed_through_step": trainer.state.global_step,
        "recovery_archives": len(list(recovery.glob("*.tar"))),
        "pinned_base_generation_config_sha256": module.file_sha(pinned_generation),
        "explicit_greedy_config": module.greedy_config(tokenizer).to_dict(),
        "script_sha256": module.file_sha(source),
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    module.finish_stage(output, [output / "verification.json"])
    assert module.stage_finished(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
