# Train the 230M rewriter

Open [rewriter_train_l4.ipynb](notebooks/rewriter_train_l4.ipynb) in Google Colab.
Choose an L4 GPU runtime. Add a read-only `HF_TOKEN` to Colab Secrets and grant the notebook
access. The token needs access to the private `jayshah5696/finetuning-labs-rewriter-training`
dataset. Run the notebook cells in order.

The notebook uses the versioned `reviewed-v3-1000/` files. It requires exactly 1,000 training,
100 development and 84 validation pairs. Each mode contributes 250 training and 25 development
pairs. The 84 validation rows must match the reviewed-v2 bytes. Validation stays outside training,
the memorization proof and checkpoint selection. The notebook pins the Hub commit on the first run
and keeps it for resume. It verifies the downloaded script against the manifest and its passing
GPT-6.1 Sol review. A release that fails these checks cannot start training.

Use a fresh runtime. Installation retains Colab's CUDA PyTorch and pins Transformers 5.21.0,
Accelerate 1.15.0, Hugging Face Hub 1.33.0, tokenizers 0.23.3, safetensors 0.8.0 and Jinja 3.1.6.
The notebook saves the installed package list. Each GPU stage also records its environment.
PyTorch and CUDA version changes are recorded between sessions. The script, dataset and pinned
training package versions must still match the original run.

## What runs

1. Check every file hash, row identity, Opus verdict, source-group boundary and token span.
   Refuse truncation. Loss covers assistant text through EOS. Prompts and padding receive no loss.
2. Save stock Liquid outputs on all 84 validation cases. Generation receives only the system
   instructions and user input. Both comparison arms use the same fully recorded greedy settings.
3. Run an optional 16-example memorization proof from the pinned base. It uses only training
   examples and saves metrics. This checks whether training can fit examples. It does not measure
   rewrite quality or select the full training checkpoint.
4. Start full training again from `LiquidAI/LFM2.5-230M` at revision
   `40cb2ad3b3044d5a41eee083a6103c8b523afa45`. Defaults are three epochs, learning rate 0.00005,
   batch size 1, gradient accumulation 16 and gradient checkpointing. FP32 parameters use BF16
   mixed precision. Every 20 optimizer steps, evaluate the 100 dev pairs and save a checkpoint.
   Select the weights with the lowest dev loss. Validation never chooses the checkpoint.
5. Save tuned outputs on the same 84 validation cases. Export a small report ZIP and retain the
   selected model directory. The notebook does not publish weights automatically.

The total context limit is 4,096 tokens. Every input reserves 2,048 output tokens. An output that
hits the limit without EOS is marked truncated and cannot count as a quality pass.

## Storage and resume

Have at least 10 GiB available on Drive and 12 GiB on local runtime disk. The script checks
filesystem free space before training. These are storage allowances, not measured L4 requirements.
Check the Drive account's quota too. Previously deleted files in Drive trash can still use it.

Training checkpoints rotate on local disk. Drive holds two fixed recovery archives, the selected
weights and reports. Each archive contains the latest full optimizer checkpoint and any different
best model's weights. Large archives are overwritten in place instead of deleting a new directory
at every save. The proof does not save model checkpoints.

To resume after a runtime reset, paste the previous `RUN_DIR` into `RESUME_RUN`. The script checks
both archive hashes and restores the newest complete archive. An interrupted copy can fall back to
the older archive. If neither is complete, a matching run restarts from the pinned base at step zero.
The log says which path it took. An interrupted proof also restarts. Completed stages verify their
saved artifact hashes and skip. Rerunning a stage cell in the same session uses the same checks.

Generation saves and hashes each case before moving on. Resume skips verified completed cases.
A partial final line is preserved separately and regenerated. Edited complete records are rejected.

## How to compare quality

The notebook's completion counter and training loss are not rewrite quality scores.
Use the same GPT-6.1 Sol rubric and settings for stock and tuned outputs. Keep candidate identity
and exact output hashes bound to each judgment. Keep the reference answer out of generation.
Human calibration remains pending, so call the model's scores diagnostic.

A whole-case pass requires correct meaning, facts, intent, uncertainty, conditions, literal text,
mode and formatting. A critical meaning change fails the case even if the prose reads well.
Report the verified pass rate for each mode and average those four rates equally. Keep every
scheduled case in its denominator. Failed, missing, uncertain and truncated outputs are not passes.
Also report long-input pass rate, unchanged-input preservation, critical meaning errors, literal
errors, truncations and judge coverage. Use the same 84 case identities for the paired comparison.
The existing [validation contract](VALIDATION.md) gives the full rubric and metric definitions.

## Implementation evidence

The CPU integration uses a tiny random 2M-parameter LFM model, the real pinned tokenizer and the
actual Trainer factory. It interrupts after step one, restores model and optimizer state from an
archive, completes step two, verifies a nonzero parameter change, and reloads the selected weights.
Its report is included in the release. This checks code paths. It does not establish L4 memory use,
BF16 execution, throughput or quality of the 230M student. Those require the actual Colab run.

To reproduce data generation and verification, use the commands in [TRAINING_DATA.md](TRAINING_DATA.md).
The notebook and runtime script require a separate exact-hash GPT-6.1 Sol implementation review. Every
released data pair also needs an exact-hash Opus input, output and pair pass. Failed and uncertain
data judgments remain evidence and are excluded. They are not human gold labels.
