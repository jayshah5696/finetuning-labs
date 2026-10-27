# Liquid and Luna diagnostic comparison

Sol assessed both models with the same rubric and hidden generator identity.
These are diagnostic scores on 68 calibration cases. Human judge calibration and
an independent test set are pending. This is not a release-quality accuracy claim.

Left: `LiquidAI/LFM2.5-230M`. Right: `openai/gpt-6-luna`.
Judge: `openai/gpt-6.1-sol` through OpenRouter and Arka.

| Slice | Stock Liquid passes | Luna passes |
| --- | ---: | ---: |
| /slack | 0/12 | 12/12 |
| /transcribe | 0/32 | 28/32 |
| /prompt | 1/12 | 11/12 |
| /polish | 1/12 | 11/12 |
| Pooled | 2/68 | 62/68 |
| Equal-mode headline | 4.2% | 92.7% |
| Short | 2/60 | 56/60 |
| Long | 0/8 | 6/8 |

| Diagnostic | Stock Liquid | Luna |
| --- | ---: | ---: |
| Definite assessment coverage | 100.0% | 100.0% |
| Semantic review coverage | 68/68 | 68/68 |
| Outputs with a judge-flagged critical error | 46 | 3 |
| Explicit identity preserved | 0/3 | 3/3 |
| Judge errors | 0 | 0 |
| Truncations | 0 | 0 |
| Uncertain | 0 | 0 |

Every scheduled case stays in the pass-rate denominator. A pass requires complete
generation, declared mechanical checks, and all five semantic criteria. Critical
errors cannot be offset by style. A missing judgment never becomes a pass.

Paired outcomes: both_pass=2, right_only_pass=60, left_only_pass=0, neither_verified_pass=6.
Luna's pooled difference is 88.2 percentage points.

The eight long cases represent four source documents. The three identity cases are
too few for a precise generalization claim. All references remain draft.

API usage is retained in comparison.json. OpenRouter reported BYOK calls. A zero
gateway charge does not mean free inference; upstream provider-reported costs are
recorded separately. GPU cost is not included in those API cost fields.
