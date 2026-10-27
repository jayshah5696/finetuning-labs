---
language:
- en
- de
- pt
license: other
license_name: private-synthetic-pilot
license_link: https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/blob/main/DATA_LICENSE.md
task_categories:
- text-generation
tags:
- text-rewriting
- synthetic
- liquid
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
- config_name: sft
  data_files:
  - split: train
    path: sft/train.jsonl
  - split: validation
    path: sft/validation.jsonl
---
# Finetuning Labs rewriter training pilot

This private synthetic pilot targets `LiquidAI/LFM2.5-230M`. GPT-6 Luna authored new fictional
documents and input variants. GPT-6.1 Sol reviewed each exported pair with the same five-criterion
rubric used in the stock Liquid and Luna diagnostic comparison. The judge is not human calibrated.
These are machine-reviewed training targets, not human gold answers.

171 training pairs and
48 internal validation pairs passed every release check.
The run scheduled 320 pairs from 80 source
documents. Rejected, uncertain, invalid, and incomplete candidates are excluded. Related modes
and variants stay in one topic group. The release has 55 source
documents in 19 topic groups; variants are not independent documents.

| Mode | Train | Internal validation |
| --- | ---: | ---: |
| /slack | 36 | 8 |
| /transcribe | 48 | 12 |
| /prompt | 43 | 12 |
| /polish | 44 | 16 |

## Simple formats

The default data has only `input`, `output`, `mode`, and `checks`. The `sft` config has only
`system`, `user`, and `assistant`, following the original local formatter guide. It uses the
same system and user prompt contract as the baseline. Do not feed `checks`, review labels,
source records, or judge critiques to the student. Supporting records stay outside supervised text.

Render the three SFT fields as one system/user/assistant conversation with the pinned Liquid
chat template. Train on assistant content through EOS only. Mask system/user tokens with -100.
The native template's newline after EOS is also masked. The preparation audit checked these spans.
No example was truncated. Use 4,096 tokens and retain the 2,048-token generation limit for evaluation.
Maximum complete training chat is 3860 tokens.

## Quality checks and boundaries

Each accepted pair preserves its protected literals, passes meaning, terms and literals, mode,
readability, and output discipline, and has no reported critical error. Uncertain is not pass.
Long inputs contain 1,000 to 2,000 pinned Liquid tokens. The data includes paragraphs, numbered
sections, and direct quotes. Identity examples teach the student to leave correct text unchanged.

The user-accepted 68-case development-validation set remains in the separate
[validation repository](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter).
Its cases, references, output fixtures, and related source families are excluded here. Exact,
embedded, and close wording overlap checks found no overlap. Those checks do not prove that all
possible semantic similarities are absent. The private personal-transcription bundle is excluded.

The internal validation split uses different topic groups and is for checkpoint selection.
The accepted 68 cases remain the fixed before/after comparison. Neither set is an unseen final
test. All per-output independent human labels remain pending. This pilot is too small to establish
production quality. It supports the next Colab L4 training proof and a measured first fine-tune.

`quality.json` has coverage and token counts. `supporting.json` binds rows to source groups,
literal rules, language/context, and judge critiques. `audit/` retains exact source/judge calls,
the rejected-candidate ledger, and the verification report. The quarantined transport rehearsal
is excluded. No training run or model improvement is claimed by this dataset publication.
