---
language: [en, de, pt]
license: other
license_name: private-reviewed-rewriter-mixed-sources
license_link: https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/blob/main/reviewed-v3-1000/DATA_LICENSE.md
task_categories: [text-generation]
configs:
- config_name: default
  data_files:
  - split: train
    path: reviewed-v3-1000/data/train.jsonl
  - split: dev
    path: reviewed-v3-1000/data/dev.jsonl
  - split: validation
    path: reviewed-v3-1000/data/validation.jsonl
- config_name: sft
  data_files:
  - split: train
    path: reviewed-v3-1000/sft/train.jsonl
  - split: dev
    path: reviewed-v3-1000/sft/dev.jsonl
  - split: validation
    path: reviewed-v3-1000/sft/validation.jsonl
---
# Reviewed rewriter training data, 1,000 examples

Use `reviewed-v3-1000/`. Training has exactly 1,000 pairs, development has 100, and validation has 84.
The previous v2 validation texts and order are unchanged. Old releases remain available;
the root 219-pair pilot is retired. There are 250 training and 25 development pairs per mode.

| Mode | Train | Dev | Validation |
| --- | ---: | ---: | ---: |
| /slack | 250 | 25 | 16 |
| /transcribe | 250 | 25 | 35 |
| /prompt | 250 | 25 | 16 |
| /polish | 250 | 25 | 17 |

Luna authors new fictional episodes through the pinned native Arka pipeline. Luna Decisions
checks the candidate input, output quality, facts, intent, literals and mode. Independent Opus
verifies each retained exact input, output and pair. Uncertain, failed, missing, malformed,
oversized and overlapping candidates are excluded. Reused v2 pairs retain their exact reviews.
Opus did not author or edit data. GPT-6.1 Sol reviews the training script and notebook;
its exact-file implementation report is in `runtime/script-review.json`.

Data rows have only input, output, mode and checks. SFT rows have only system, user and assistant.
Context needed for rewriting enters the user request; all other metadata stays outside loss.
All chats fit 4,096 tokens without truncation and reserve 2,048 tokens for generation. Long
examples, quotations, numbered sections and unchanged examples remain included. See quality.json
for measured coverage. Source groups do not cross splits. Training has 1,000 distinct targets.

The notebook in `notebooks/rewriter_train_l4.ipynb` loads the private Hub files and pinned
LiquidAI/LFM2.5-230M weights. Use an L4 runtime and a read-only HF_TOKEN in Colab Secrets.
It checks data, saves stock outputs, runs an optional memorization proof, trains on train with
checkpoint selection from dev, and saves tuned outputs on the frozen validation set.
It uses Transformers full fine-tuning and assistant-only labels through EOS. Checkpoints and
reports go to Drive by default. Checkpoints rotate on local disk; two fixed verified recovery
archives persist on Drive. The proof saves metrics only. Have 10 GiB free on Drive and 12 GiB
on local disk before training. Completed stages skip after verifying their artifact hashes.
It does not publish weights automatically. The actual 230M
student has not been trained as part of this data preparation.

These are model judgments, not human gold. Synthetic dictation is not recorded ASR evidence.
Real user preferences, independent human calibration and an unseen final test remain pending.
Preserve TypeWhisper's GPL v3 source snapshot and notice in validation-bundle/supporting/.
No personal transcriptions, recordings or credentials are included. All publication is private.
