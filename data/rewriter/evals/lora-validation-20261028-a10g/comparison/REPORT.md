# Rewriter diagnostic comparison

Sol assessed both models with the same rubric and hidden generator identity.
These are diagnostic scores on 84 validation cases. Human judge calibration and
an independent test set are pending. This is not a release-quality accuracy claim.

Left: `LiquidAI/LFM2.5-230M`. Right: `jayshah5696/lfm2.5-230m-rewriter-lora`.
Judge: `openai/gpt-6.1-sol` through OpenRouter and Arka.

| Slice | Stock Liquid passes | Compared model passes |
| --- | ---: | ---: |
| /slack | 0/16 | 14/16 |
| /transcribe | 0/35 | 19/35 |
| /prompt | 4/16 | 6/16 |
| /polish | 0/17 | 9/17 |
| Pooled | 4/84 | 48/84 |
| Equal-mode headline | 6.2% | 58.1% |
| Short | 4/71 | 45/71 |
| Long | 0/13 | 3/13 |

| Diagnostic | Stock Liquid | Compared model |
| --- | ---: | ---: |
| Definite assessment coverage | 100.0% | 100.0% |
| Semantic review coverage | 84/84 | 84/84 |
| Outputs with a judge-flagged critical error | 33 | 25 |
| Explicit identity preserved | 0/8 | 8/8 |
| Judge errors | 0 | 0 |
| Truncations | 0 | 0 |
| Uncertain | 0 | 0 |

Every scheduled case stays in the pass-rate denominator. A pass requires complete
generation, declared mechanical checks, and all five semantic criteria. Critical
errors cannot be offset by style. A missing judgment never becomes a pass.

Paired outcomes: both_pass=2, right_only_pass=46, left_only_pass=2, neither_verified_pass=34.
The compared model's pooled difference is 52.4 percentage points.

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
| meaning | 52/84 | 60/84 |
| terms_and_literals | 72/84 | 71/84 |
| mode | 5/84 | 49/84 |
| readability | 71/84 | 78/84 |
| output_discipline | 8/84 | 83/84 |
