# Status

## Current work

The rewriter calibration draft is implemented for review. It has 68 cases, four modes,
156 authored output fixtures, a simple review page, and the guide's three-field export.
Source context and validation rules remain in supporting files.

Focused failure tests were written before production code. The first test run failed because the
package did not exist. The long-document failure tests also preceded their implementation.
All 34 tests pass on Python 3.12. Ruff checks and formatting pass.
All authored mechanical fixture expectations match. Forty-nine deliberately wrong outputs still
pass term checks and require semantic review; this is an explicit validation boundary.

`make review` generated the four-column review page. Desktop and narrow-screen previews had no
console errors. The expanded export has 68 guide-format references with only system, user, and
assistant fields, saved at `artifacts/benchmark-guide-68.jsonl`. The original 60-row export is
preserved. A real overwrite attempt failed and preserved the existing export. Required
dictionary and profile context survive export. A changed code identifier is rejected.

`make build` produced the source distribution and wheel. An isolated wheel environment outside
the repository audited all 68 cases and exported all 68 references without the TypeWhisper
checkout. The imported source snapshot matches its original SHA-256.

Private review page: https://sangam.jay-hirajoshi.ts.net/p/finetuning-labs-benchmark-draft
Sangam document: `a43249ab-3ed4-4055-97ef-f646b1710441`.
Benchmark revision: `92b2712f-b807-4c8b-987d-728c1e3a062d`, publication version 2, private.
Plan revision: `bf9cac8a-6309-43e1-b8ce-2d5881c48f40`, publication version 3, private.
Both stored documents match their local HTML exactly.

## Personal sample

After the user authorized choosing any source, 10 actual history entries were selected read-only.
Their original raw text, historical app output, and proposed corrections stay in an ignored local
bundle. Historical cleanup output is not treated as a preferred correction. All 10 new references
remain pending human review. The shared 68-case publication contains no personal text.
The private 10-case bundle and its 21 output fixtures pass the offline audit. It has a local
four-column review page. Actual wheel and source-archive contents were checked for the selected
raw text and source IDs; neither archive contains them. Git ignores the private bundle. The
history file's SHA-256 remained unchanged during selection and preparation.

## Pending

- User review of the proposed references and checks. All 68 worksheet rows are pending.
- User review of the 10 personal reference drafts and any preferred corrections.
- Independent development and locked-test data.
- GPT-6.1 Sol validator calibration and real Luna/Liquid baseline calls.
- Colab L4 notebook, 16-row training proof, and the fine-tuned comparison.
- Hugging Face packaging and publication after the data is reviewed.

No paid model calls, student training, benchmark freeze, or cited-lab migration occurred.

## Long-document addition, 27 October 2026

Eight authored examples were appended, two per mode, from four fictional source families.
Every mode has one input near 1,100 tokens and another near 1,850 tokens. The documents include
long paragraphs, numbered sections, and direct quotes. Two also contain Markdown block quotes;
one has indented bullets. All first 60 rows, their supporting records, and their 132 fixtures
were verified unchanged. Related variants share a source group and must stay together in a split.

The native Liquid tokenizer at revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45` measured
1,094 to 1,865 input tokens. The longest complete reference chat has 3,877 tokens. With a
2,048-token generation allowance, the largest prompt plus allowance has 4,065 tokens. The draft
fits a 4,096-token context. The earlier 1,024-token setting is no longer suitable. No text was
truncated. These lengths do not establish L4 memory use or model quality.

`make tokens` reproduces the length check with the optional locked data dependencies. Saved
long-case evidence is in `supporting/long-form-tokens.json`. An independent check serialized the
actual exported chats as text and encoded them again. All eight complete lengths matched, every
quote survived, and all numbered sections remained. The review page links to all eight long
cases and keeps their full texts in expandable cells.

The 24 new fixtures contain eight reference outputs and 16 wrong outputs. Thirteen new wrong
outputs pass literal checks. They test truncation, changed quotes, reversed amounts, invented
approval, and instructions inside quoted text. Their semantic labels remain authored expectations.

## Validation definition and stopping point

The validation contract now defines a whole-case pass, exact metric denominators, judge checks,
and the stock-versus-tuned comparison. The primary result is the verified pass rate in each
mode. The headline averages the four mode rates equally. The pooled rate retains all scheduled
cases in its denominator. Uncertain, missing, truncated, and failed outputs do not count as
verified passes. Decided-case pass rate is a separate diagnostic shown beside coverage.

Human calibration remains pending. It should include the eight long references and all 16 long
wrong outputs. The user subsequently authorized diagnostic model runs, a private Hub upload,
and Git commits. The runner and metric aggregation are now implemented. Their scores remain
advisory until human calibration. The current runtime execution status will be recorded below.

## Publication and baseline execution

The historical long-example authoring script, an allowlist-only Hub packager, durable baseline
and judge runners, and an L4 Modal entry point are implemented. Twelve new failure tests preceded
the runtime code. The initial authoring script reproduces the exact eight committed long records.
Arka uses clean commit `779040204b0f392465b7e6240bf9d9200816dc5d`; unrelated edits in the user's
Arka checkout are not included. Hosted calls use OpenRouter, not a direct OpenAI key.

Hugging Face and Modal authentication work. The active zshrc has no OpenRouter key and its optional
local file is absent. The user has been asked only for the correct source file path. Shared-data
publication and Liquid execution can proceed independently. No training is authorized here.
