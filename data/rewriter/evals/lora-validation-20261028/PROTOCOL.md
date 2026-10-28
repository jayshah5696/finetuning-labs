# Stock Liquid and trained LoRA validation

This diagnostic compares the pinned stock model with the uploaded LoRA on all 84 frozen
reviewed-v2 validation cases. It does not change the dataset or select a new checkpoint.

## Frozen inputs

- Base model: LiquidAI/LFM2.5-230M at `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
- LoRA: jayshah5696/lfm2.5-230m-rewriter-lora at `034d54e5b7c291fea1f8fcacba77fc8dad543bc4`.
- Public dataset: jayshah5696/finetuning-labs-rewriter-training, SFT validation, at
  `bbcdb7692445a4310296060606b8a6dcce439f47`.
- Local case context and literal rules: `data/rewriter/reviewed-v2/validation-bundle`.

Before submission, the runner compared every SFT row from the public Parquet file with the
messages and references reconstructed from the local bundle. All 84 matched. The run receipt
records the Parquet hash and full bundle identity.

The split contains 16 Slack, 35 transcribe, 16 prompt and 17 polish cases. Thirteen are long
cases. Related source groups are preserved. References are never sent to generation.

## Generation

Both arms run on the same Modal L4, using BF16, eager attention, the pinned base tokenizer,
seed 42, greedy decoding, one beam, a 4,096-token context and at most 2,048 generated tokens.
The runner installs one explicit GenerationConfig rather than inheriting model-specific
decoding options. Both generation records retain the same configured decoding object and
library versions. Transformers fills null fields with its pinned library defaults during
generation. The model's generation configuration is replaced with the shared object first,
so model-specific defaults cannot differ between arms.
The adapter is the only difference between arm configurations. Comparison rejects other
configuration differences. This run does not measure optimized desktop latency.

Each case has one durable request and response record. Interrupted or failed attempts remain
visible and are not silently reissued. A response that reaches its token limit without EOS is truncated,
even if the retained portion reads well. A normal EOS does not establish semantic completeness.

## Assessment

Both arms use the unchanged judge rubric in validation-bundle/supporting/judge-system.txt
and the judge settings from configs/rewriter.toml: openai/gpt-6.1-sol through the pinned Arka
OpenRouter client, high reasoning effort and a 16,384-token response allowance.
The judge receives input, context, mode, checks, reference and candidate, without model identity.

A whole-case pass requires completed generation, all declared mechanical checks and all five
semantic criteria. Meaning, terms and literals, mode, readability and output discipline must
all pass. A critical error fails the case. Missing, failed, truncated or uncertain results
stay in the denominator and do not count as passes.

Report pass counts and rates by mode, their equal-weight average, pooled passes, long-case
results, identity preservation, critical errors and paired wins and regressions. Recompute
metrics from the bound raw calls. Do not accept editable summary files as score evidence.

The setup Sol control is separate from the 168 scheduled candidate assessments. API usage
and GPU execution are separate records. Gateway and upstream costs, when returned, are separate.

Human judge calibration and an independent final test remain pending. This split was used
during project development. The scores are model judgments on development validation.
