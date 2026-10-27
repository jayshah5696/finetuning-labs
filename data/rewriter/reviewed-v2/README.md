---
language:
- en
- de
- pt
license: other
license_name: private-reviewed-rewriter-mixed-sources
license_link: https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/blob/main/reviewed-v2/DATA_LICENSE.md
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
    path: reviewed-v2/data/train.jsonl
  - split: dev
    path: reviewed-v2/data/dev.jsonl
  - split: validation
    path: reviewed-v2/data/validation.jsonl
- config_name: sft
  data_files:
  - split: train
    path: reviewed-v2/sft/train.jsonl
  - split: dev
    path: reviewed-v2/sft/dev.jsonl
  - split: validation
    path: reviewed-v2/sft/validation.jsonl
---
# Reviewed rewriter data v2

Use `reviewed-v2/`. The original 219-pair pilot in the repository root failed independent
Opus data verification and is retired for training. Its old files remain for reproduction.

This version has 65 training, 17 development and 84 validation pairs for
`LiquidAI/LFM2.5-230M`. Each released input, output and pair passed an exact-row Opus 5.5
review. Two unresolved rows are excluded. These are model judgments, not human gold labels.

| Mode | Train | Dev | Validation |
| --- | ---: | ---: | ---: |
| /slack | 17 | 4 | 16 |
| /transcribe | 17 | 4 | 35 |
| /prompt | 15 | 5 | 16 |
| /polish | 16 | 4 | 17 |

## Sources and formats

GPT-6 Luna generated one fictional pair per independent scenario through clean Arka commit
`5d353740ae1dd34742841071752814028db1489a`. The native pipeline ran generation, normalization, exact and length checks,
`decision_panel`, and `labeling_engine` with Luna Decisions. The parent corrected defects from
data-only Opus reviews and resubmitted every changed row. Opus did not generate or edit data.

The 84 validation pairs contain all 68 original benchmark cases, with two versioned corrections,
plus 16 new pairs. Thirty original cases derive from TypeWhisper authored fixtures. The rest
are authored or synthetic. No personal transcriptions or recordings are included. Preserve the
TypeWhisper GPL v3 notice and source snapshot supplied in `validation-bundle/supporting/`.
The original 68-case files and their historical model scores remain unchanged elsewhere.

Default rows have only `input`, `output`, `mode` and `checks`. SFT rows have only `system`,
`user` and `assistant`, as in the supplied formatter guide. Scores, critiques, IDs and source
records are separate. Only required language, profile and dictionary context enters the request.

## Quality evidence

All 168 candidate pairs were tested with six native decision dimensions. Arka automatically
accepted 75. The independent Opus reviews adjudicate the other candidates; native scores and
policy outcomes remain visible in `supporting.json`. A released row does not imply a native
automatic pass. Probabilities have not been checked against independent human labels.

The 166 retained pairs include 33 long inputs of 1,094 to 1,906 pinned Liquid tokens. Unchanged
examples occur at short, medium and long lengths. Training has 65 distinct source groups and
65 distinct targets. No source group crosses any of the three splits. Exact, embedded and
near-copy checks found no overlap between training and the original benchmark, its fixtures,
or the local personal holdout. Wording checks do not prove semantic independence.

Every complete chat fits 4,096 tokens, and every prompt leaves room for 2,048 generated tokens.
The maximum complete chat is 4029 tokens. No example was truncated.
Train on assistant content through EOS only. Mask system/user tokens and the template's trailing
newline with -100. The Colab dataset-check notebook verifies these spans without loading weights.

Use `dev` for checkpoint selection and `validation` for the fixed stock-versus-tuned comparison.
The latter has 80 source groups and 80 distinct targets because some original long cases are
related mode variants. This is development validation, not an unseen final test. Rerun the stock
baseline on v2 before comparing it with a fine-tune; historical v1 scores are not v2 scores.

The dataset is small and predominantly English. Inputs are synthetic drafts and dictation,
not measured ASR output from recordings. Dictionary and profile coverage remain limited.
Uncertainty-preservation cases are overrepresented. No student training or quality improvement
is claimed. Human calibration and broader real-use evaluation remain pending.

`quality.json` records counts and limits. `reviews/` contains exact verdicts and the report-recovery
evidence. `evidence/` retains source plans, generation attempts, corrections, rejected candidates,
native records and pipeline statistics. These evidence files are not training inputs.
