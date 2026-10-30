# Rewriter diagnostic comparison

Sol assessed both models with the same rubric and hidden generator identity.
These are diagnostic scores on 81 validation cases. Human judge calibration and
an independent test set are pending. This is not a release-quality accuracy claim.

Left: `LiquidAI/LFM2.5-350M`. Right: `jayshah5696/lfm2.5-350m-rewriter-lora`.
Judge: `openai/gpt-6.1-sol` through OpenRouter and Arka.

| Slice | Stock Liquid passes | Compared model passes |
| --- | ---: | ---: |
| /slack | 5/16 | 15/16 |
| /transcribe | 3/32 | 24/32 |
| /prompt | 4/16 | 11/16 |
| /polish | 6/17 | 12/17 |
| Pooled | 18/81 | 62/81 |
| Equal-mode headline | 25.2% | 77.0% |
| Short | 18/68 | 55/68 |
| Long | 0/13 | 7/13 |

| Diagnostic | Stock Liquid | Compared model |
| --- | ---: | ---: |
| Definite assessment coverage | 100.0% | 100.0% |
| Semantic review coverage | 81/81 | 81/81 |
| Outputs with a judge-flagged critical error | 50 | 13 |
| Explicit identity preserved | 2/8 | 7/8 |
| Judge errors | 0 | 0 |
| Truncations | 0 | 0 |
| Uncertain | 0 | 0 |

Every scheduled case stays in the pass-rate denominator. A pass requires complete
generation, declared mechanical checks, and all five semantic criteria. Critical
errors cannot be offset by style. A missing judgment never becomes a pass.

Paired outcomes: both_pass=18, right_only_pass=44, left_only_pass=0, neither_verified_pass=19.
The compared model's pooled difference is 54.3 percentage points.

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
| meaning | 34/81 | 69/81 |
| terms_and_literals | 51/81 | 76/81 |
| mode | 19/81 | 66/81 |
| readability | 65/81 | 77/81 |
| output_discipline | 40/81 | 81/81 |
