# Results history

This page records the progression of diagnostic evaluations on the English and multilingual development-validation sets. All semantic scores are LLM-judge diagnostics from `openai/gpt-6.1-sol` on development validation, not independent human gold labels or results on an unseen final test.

For the full chronological project log, see [status.md](status.md) and [english-rewriter-experiment-log.html](english-rewriter-experiment-log.html).

## Latest five-model English evaluation (1 November 2026, 141 cases)

Both rank-16 LoRA adapters were trained on the 1,979-row English training dataset at Hugging Face commit `36a75c009d008415fad557d49005ba4e5c73f4d7`. The 230M adapter is pinned to commit `664f541` (`664f541045c0148e7bec66b267b35238716d4688`) and the 350M adapter is pinned to commit `b660a20` (`b660a2046092b07e46061ee25b738da3f4ef0688`). All five arms generated fresh outputs on all 141 English validation cases (81 original frozen cases plus 60 expansion cases) using the same prompts, 2,048-token output budget, and `openai/gpt-6.1-sol` judge rubric.

| Model | Arm | All 141 strict passes | Original 81 passes | Expansion 60 passes | Critical-error cases (141) | Truncated (141) | Identity unchanged | Long passes (141) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LiquidAI/LFM2.5-230M` | base | 5/141 (3.5%) | 4/81 (4.9%) | 1/60 (1.7%) | 72 | 0 | 0/8 | 0/25 |
| `LiquidAI/LFM2.5-230M` | LoRA | 64/141 (45.4%) | 53/81 (65.4%) | 11/60 (18.3%) | 60 | 1 | 7/8 | 4/25 |
| `LiquidAI/LFM2.5-350M` | base | 16/141 (11.3%) | 16/81 (19.8%) | 0/60 (0.0%) | 99 | 0 | 2/8 | 0/25 |
| `LiquidAI/LFM2.5-350M` | LoRA | 80/141 (56.7%) | 63/81 (77.8%) | 17/60 (28.3%) | 49 | 0 | 7/8 | 7/25 |
| `openai/gpt-6-luna` | cloud | 95/141 (67.4%) | 70/81 (86.4%) | 25/60 (41.7%) | 27 | 0 | 4/8 | 9/25 |

- The 230M LoRA has one truncated output after repeating `--next`; it counts as a failure and has no semantic judgment. The other 704 outputs have complete reviews with no judge errors or uncertain verdicts.
- None of the five models passes any of the 12 new long stress cases in the 60-case expansion under the current strict rules.
- Full artifacts: [data/rewriter/evals/english-five-models-20261101/README.md](../data/rewriter/evals/english-five-models-20261101/README.md), [comparison.json](../data/rewriter/evals/english-five-models-20261101/comparison.json), and [case-results.csv](../data/rewriter/evals/english-five-models-20261101/case-results.csv).

## Five-model English comparison on the original 81 cases (29 October 2026)

Before the 60-case benchmark expansion and the 1,979-row retraining run, all five models were evaluated on the 81 frozen English validation cases. In that run, the 350M LoRA recorded the 1,864-row English training data, while the 230M LoRA still recorded the earlier 1,500-row training data (`ba71856780e4d2398218fe76756b3390556437ea`).

| Model | Arm | Passes (81) | Critical errors | Truncated | Identity unchanged | Long passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `LiquidAI/LFM2.5-230M` | stock | 4/81 (4.9%) | 33 | 0 | 0/8 | 0/13 |
| `LiquidAI/LFM2.5-230M` | LoRA (1,500-row) | 50/81 (61.7%) | 17 | 1 | 7/8 | 4/13 |
| `LiquidAI/LFM2.5-350M` | stock | 18/81 (22.2%) | 50 | 0 | 2/8 | 0/13 |
| `LiquidAI/LFM2.5-350M` | LoRA (1,864-row) | 62/81 (76.5%) | 13 | 0 | 7/8 | 7/13 |
| `openai/gpt-6-luna` | cloud | 73/81 (90.1%) | 1 | 0 | 4/8 | 10/13 |

- Per-mode passes for Luna: `/slack` 16/16, `/transcribe` 27/32, `/prompt` 15/16, `/polish` 15/17.
- Per-mode passes for 350M LoRA: `/slack` 15/16, `/transcribe` 24/32, `/prompt` 11/16, `/polish` 12/17.
- Per-mode passes for 230M LoRA: `/slack` 15/16, `/transcribe` 18/32, `/prompt` 7/16, `/polish` 10/17.
- Full artifacts: [five-model report](../data/rewriter/evals/english-all-models-20261029/REPORT.md), [case-reviews.csv](../data/rewriter/evals/english-all-models-20261029/case-reviews.csv), and [four-arm Liquid report](../data/rewriter/evals/english-comparison-20261029/REPORT.md).

## Earlier 230M evaluations on the 84-case multilingual validation set (28-29 October 2026)

On the frozen 84-case reviewed-v2 validation set (81 English cases plus 3 German/Portuguese cases):

- **1,500-example 230M LoRA (`ba71856780e4d2398218fe76756b3390556437ea`):** Stock 230M passed 4/84 (equal-mode 6.25%) and the updated 230M LoRA passed 53/84 (equal-mode 64.1%, 51/81 on the English subset under that run's cached judgment). Long passes were 4/13, identity was 7/8, with 18 critical-error outputs among 83 complete outputs and 1 truncation. See [FINDINGS.md](../data/rewriter/evals/lora-1500-validation-20261029-a10g/FINDINGS.md), [REPORT.md](../data/rewriter/evals/lora-1500-validation-20261029-a10g/comparison/REPORT.md), and [ENGLISH.md](../data/rewriter/evals/lora-1500-validation-20261029-a10g/ENGLISH.md).
- **1,000-example 230M LoRA (`034d54e5b7c291fea1f8fcacba77fc8dad543bc4`):** Stock 230M passed 4/84 and the first 230M LoRA passed 48/84 (equal-mode 58.1%, 46/81 on the English subset). Identity improved from 0/8 to 8/8, long passes from 0/13 to 3/13, and critical errors dropped from 33 to 25. See [REPORT.md](../data/rewriter/evals/lora-validation-20261028-a10g/comparison/REPORT.md) and [FINDINGS.md](../data/rewriter/evals/lora-validation-20261028-a10g/FINDINGS.md).

## Original 68-case calibration baseline (27 October 2026)

On the initial 68-case calibration benchmark:

- Stock `LiquidAI/LFM2.5-230M` passed 53/68 deterministic mechanical checks, 0/8 long cases, and 0/3 explicit identity cases ([mechanical report](../data/rewriter/evals/baseline-20261027/liquid-mechanical/REPORT.md)).
- Under matched `openai/gpt-6.1-sol` semantic judging, stock `LiquidAI/LFM2.5-230M` passed 2/68 cases (4.2% equal-mode headline, 46 critical errors) and `openai/gpt-6-luna` passed 62/68 cases (92.7% equal-mode headline, 3 critical errors). See [baseline comparison report](../data/rewriter/evals/baseline-20261027/comparison/REPORT.md).
