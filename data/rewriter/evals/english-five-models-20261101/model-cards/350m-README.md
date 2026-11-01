---
library_name: peft
base_model: LiquidAI/LFM2.5-350M
datasets:
- jayshah5696/finetuning-labs-rewriter-training
tags: [lora, text-rewriting]
pipeline_tag: text-generation
---
# Rewriter LoRA

LoRA adapter for `LiquidAI/LFM2.5-350M` at revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
Training data: `jayshah5696/finetuning-labs-rewriter-training` / `sft` at `36a75c009d008415fad557d49005ba4e5c73f4d7`.
Train rows: 1979. Dev rows: 89.
Rank: 16. Alpha: 32.
The checkpoint was selected by dev loss. Final validation loss: 0.305981.
This loss does not establish semantic rewrite quality. The original base model is required.
The base model's license and dataset source notices still apply.
