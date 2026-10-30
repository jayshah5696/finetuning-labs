# Rewriter diagnostic comparison

Sol assessed both models with the same rubric and hidden generator identity.
These are diagnostic scores on 81 validation cases. Human judge calibration and
an independent test set are pending. This is not a release-quality accuracy claim.

Left: `LiquidAI/LFM2.5-230M`. Right: `jayshah5696/lfm2.5-230m-rewriter-lora`.
Judge: `openai/gpt-6.1-sol` through OpenRouter and Arka.

| Slice | Stock Liquid passes | Compared model passes |
| --- | ---: | ---: |
| /slack | 0/16 | 15/16 |
| /transcribe | 0/32 | 18/32 |
| /prompt | 4/16 | 7/16 |
| /polish | 0/17 | 10/17 |
| Pooled | 4/81 | 50/81 |
| Equal-mode headline | 6.2% | 63.1% |
| Short | 4/68 | 46/68 |
| Long | 0/13 | 4/13 |

| Diagnostic | Stock Liquid | Compared model |
| --- | ---: | ---: |
| Definite assessment coverage | 100.0% | 98.8% |
| Semantic review coverage | 81/81 | 80/81 |
| Outputs with a judge-flagged critical error | 33 | 17 |
| Explicit identity preserved | 0/8 | 7/8 |
| Judge errors | 0 | 0 |
| Truncations | 0 | 1 |
| Uncertain | 0 | 0 |

Every scheduled case stays in the pass-rate denominator. A pass requires complete
generation, declared mechanical checks, and all five semantic criteria. Critical
errors cannot be offset by style. A missing judgment never becomes a pass.

Paired outcomes: both_pass=3, right_only_pass=47, left_only_pass=1, neither_verified_pass=30.
The compared model's pooled difference is 56.8 percentage points.

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
| meaning | 50/81 | 63/81 |
| terms_and_literals | 69/81 | 72/81 |
| mode | 5/81 | 55/81 |
| readability | 69/81 | 73/81 |
| output_discipline | 8/81 | 79/81 |
