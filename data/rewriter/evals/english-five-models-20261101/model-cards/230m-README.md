---
library_name: peft
base_model: LiquidAI/LFM2.5-230M
datasets:
- jayshah5696/finetuning-labs-rewriter-training
tags: [lora, text-rewriting]
pipeline_tag: text-generation
---
# Rewriter LoRA

LoRA adapter for `LiquidAI/LFM2.5-230M` at revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
Training data: `jayshah5696/finetuning-labs-rewriter-training` / `sft` at `36a75c009d008415fad557d49005ba4e5c73f4d7`.
Train rows: 1979. Dev rows: 89.
Rank: 16. Alpha: 32.
The checkpoint was selected by dev loss. Final validation loss: 0.361189.
This loss does not establish semantic rewrite quality. The original base model is required.
The base model's license and dataset source notices still apply.
