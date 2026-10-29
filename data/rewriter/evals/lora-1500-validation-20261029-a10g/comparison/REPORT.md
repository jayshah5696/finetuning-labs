# Rewriter diagnostic comparison

Sol assessed both models with the same rubric and hidden generator identity.
These are diagnostic scores on 84 validation cases. Human judge calibration and
an independent test set are pending. This is not a release-quality accuracy claim.

Left: `LiquidAI/LFM2.5-230M`. Right: `jayshah5696/lfm2.5-230m-rewriter-lora`.
Judge: `openai/gpt-6.1-sol` through OpenRouter and Arka.

| Slice | Stock Liquid passes | Compared model passes |
| --- | ---: | ---: |
| /slack | 0/16 | 15/16 |
| /transcribe | 0/35 | 21/35 |
| /prompt | 4/16 | 7/16 |
| /polish | 0/17 | 10/17 |
| Pooled | 4/84 | 53/84 |
| Equal-mode headline | 6.2% | 64.1% |
| Short | 4/71 | 49/71 |
| Long | 0/13 | 4/13 |

| Diagnostic | Stock Liquid | Compared model |
| --- | ---: | ---: |
| Definite assessment coverage | 100.0% | 98.8% |
| Semantic review coverage | 84/84 | 83/84 |
| Outputs with a judge-flagged critical error | 33 | 18 |
| Explicit identity preserved | 0/8 | 7/8 |
| Judge errors | 0 | 0 |
| Truncations | 0 | 1 |
| Uncertain | 0 | 0 |

Every scheduled case stays in the pass-rate denominator. A pass requires complete
generation, declared mechanical checks, and all five semantic criteria. Critical
errors cannot be offset by style. A missing judgment never becomes a pass.

Paired outcomes: both_pass=3, right_only_pass=50, left_only_pass=1, neither_verified_pass=30.
The compared model's pooled difference is 58.3 percentage points.

The long slice contains 13 cases.
There are 8 explicit identity cases.
The validation set is small and was used during development. It is not an unseen final test.

API usage is retained in comparison.json. A zero gateway charge does not mean
free inference; upstream provider-reported costs, when available, are
recorded separately. GPU cost is not included in those API cost fields.

## Semantic criteria

These counts describe each criterion separately. They do not replace the whole-case
pass rule. A case can fail several criteria. Missing assessments never count as passes.

| Criterion | Stock Liquid passes | Compared model passes |
| --- | ---: | ---: |
| meaning | 52/84 | 65/84 |
| terms_and_literals | 72/84 | 74/84 |
| mode | 5/84 | 58/84 |
| readability | 71/84 | 75/84 |
| output_discipline | 8/84 | 82/84 |
