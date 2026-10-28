# Status

## Reviewed v3, 1,000 training examples

The Colab L4 training notebook and standalone runtime are implemented. The first Opus code
review failed on checkpoint storage, interruption recovery, repeated completed stages and
inherited decoding settings. The parent fixed those findings. The next Opus call hit a provider
credit limit. The user changed the remaining review and fix loop to GPT-6.1 Sol only. Sol's exact
script and notebook review passed with no unresolved material findings. The first failed report
remains in the release history. Sol did not author or edit examples or implementation.

All 91 focused tests, Ruff checks and formatting pass. The package builds. A tiny random
2M-parameter LFM completed a CPU interruption/recovery integration with the real pinned
tokenizer and actual Trainer factory. It stopped after step one, restored model and optimizer
state, completed step two, showed a nonzero parameter update and reloaded its selected weights.
The report binds the exact runtime script. No 230M student training, L4 or BF16 run occurred.

Native Arka and Luna generated 1,440 main candidates, 200 supplemental candidates and 80 extra
prompt candidates. Extra scenarios compensate for rejected data without changing acceptance
criteria. Each mechanically valid candidate receives six native Decisions scores and independent
Opus input, output and pair verification. Failed and uncertain verdicts stay excluded. Interrupted
or malformed review calls remain preserved; explicit resume retries only incomplete verification.
A repeated wrong-hash batch was recovered through individually constrained review packets.
The 14 remaining incomplete reviews are excluded. No new Opus call followed the user's change.

The final export has exactly 1,000 train and 100 dev passes, with 250 and 25 per mode, and the
unchanged 84 reviewed-v2 validation rows. Training has 1,000 distinct targets and source groups.
All retained exact inputs, outputs and pairs have completed Opus passes. Of the new candidates,
1,648 received completed reviews: 1,330 pass, 241 fail and 77 uncertain. Schema, context or length
checks rejected 57 candidates; another failed a mechanical overlap check. The final quotas leave
312 passing candidates unused. The release retains 65 train, 17 dev and all 84 validation rows
from v2, adding 935 train and 83 dev rows.

There are 139 long training inputs, 14 long dev inputs and 13 long validation inputs, from
1,004 to 1,906 Liquid tokens. All long training targets have paragraph breaks, 92 have numbered
sections and 116 contain quoted text. Training includes 188 unchanged pairs. The selected new
training rows cover 68 declared domains. No eight-word target sequence occurs in more than
five of the 1,000 training documents. These checks describe distribution, not semantic accuracy.

The private Hub revision is `f3e97879e6a8006e5dd8fedf2c99379c0d097671` under `reviewed-v3-1000/`.
All 331 upload files were downloaded and verified byte for byte. All 397 historical files other
than the intentionally replaced root README retain their Hub blob IDs. All original benchmark
and v2 validation bytes are preserved. The receipt and preservation audit are committed under
`data/rewriter/releases/`.

The actual uploaded notebook's download and readiness cells ran locally with cached Hub auth
and local storage. They verified the approved script, Sol report, all 1,184 rows, all loss masks,
all source groups, the frozen validation hashes and the 16-row proof selection. No GPU stage ran.
The notebook is ready for the user's L4 run. The memorization proof, full 230M training and paired
stock/tuned semantic comparison remain the next work. Human labels and a locked final test remain
pending. The review-time preparation context is retained without duplicating its data files.

## Reviewed data v2, 27 October 2026

The original 219-pair pilot failed independent Opus data verification and is retired for training.
Its data files and receipts remain historical. The replacement has 65 training, 17 development
and 84 validation pairs. Every exact released input, output and pair has an Opus 5.5 pass.
One failed prompt and one uncertain transcription are excluded. Opus was restricted to data
verification. It did not generate or edit examples or review implementation.

Clean Arka commit `5d353740ae1dd34742841071752814028db1489a` supplies native generation,
decision-panel and labeling stages. The final native check scored 168 pairs and automatically
accepted 75. Independent Opus passes adjudicate the retained disagreements. Scores, policy
outcomes, critiques and hashes remain separate from supervised text. The user's dirty Arka
checkout was not modified.

All 166 pairs fit the pinned Liquid chat template without truncation. There are 33 long inputs
of 1,094 to 1,906 tokens and unchanged cases at short, medium and long lengths. The complete
chat maximum is 4,029 tokens. Source families do not cross any of the three splits. Training
contains 65 distinct targets from 65 new fictional scenarios. No personal history is shared.

The original 68-case benchmark and historical scores remain unchanged. V2 validation includes
all 68 cases with two versioned corrections and 16 new pairs. The runnable 84-case bundle is
`data/rewriter/reviewed-v2/validation-bundle`. Baselines must be rerun on v2 before a tuned
comparison. Human labels, an unseen final test and student training remain pending.

The new Colab entry point is `notebooks/rewriter_reviewed_data_check.ipynb`. It checks the data
and selects 16 training rows. Actual Colab execution and the L4 training proof remain pending.
The sections below record earlier work and superseded release claims.

The private Hub revision is `70715437e3d6b67ef47eb89c83ccb1a66038082e`. All 69 uploaded files
were downloaded and verified byte for byte. Its default configs use `reviewed-v2/`. The three
notebook data cells ran locally against this actual Hub revision. All 166 reviews and token
spans, the split boundaries and the 16-row proof selection passed. Colab Secrets was replaced
by cached Hub authentication for that local run. No GPU execution is claimed.

All 65 focused tests, Ruff checks and formatting pass. The package builds. Re-export from the
committed pool and review evidence produces all six data and SFT files byte for byte. The four
original validation-freeze hashes remain unchanged.

## Current work

On 27 October 2026 the user accepted the 68 cases for development validation and authorized
pushing the project changes. `data/rewriter/validation-freeze.json` records the acceptance and
file hashes. The benchmark bytes and observed failures remain unchanged. All 64 source families
and all fixtures stay outside training. Per-output human labels and judge calibration remain
pending. The next work is a separate training-data pilot for the same four rewrite modes.

The rewriter calibration draft is implemented for review. It has 68 cases, four modes,
156 authored output fixtures, a simple review page, and the guide's three-field export.
Source context and validation rules remain in supporting files.

Focused failure tests were written before production code. The first test run failed because the
package did not exist. The long-document failure tests also preceded their implementation.
All 49 tests pass on Python 3.12. Ruff checks and formatting pass.
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
Plan revision: `c7d18d71-2527-4763-a882-00fa1fbaa3a1`, publication version 5, private.
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

- Independent per-output human labels. The user accepted the dataset for development validation;
  the 68-row human worksheet remains pending and is not fabricated from that acceptance.
- User review of the 10 personal reference drafts and any preferred corrections.
- An independent locked final test. The training pilot has its own internal development split.
- Independent human calibration of GPT-6.1 Sol. Both model baselines and Sol judging are complete.
- Execution of the supplied Colab L4 notebook, 16-row training proof, and fine-tuned comparison.
- Any public Hugging Face release after data and license review. The private draft is uploaded.

The authorized model baselines and shared judging are complete. Development validation is now
frozen. No student training or cited-lab migration occurred. All model scores remain diagnostic.

## Training-data pilot, 27 October 2026

The final preparation run is `data/rewriter/training/v3-20261027`. It retained 219 pairs from
320 planned pairs. Training has 171 rows from 43 source documents in 15 topic groups. Internal
validation has 48 rows from 12 documents in four different topic groups. The four modes are
present in both splits. Related variants and modes stay in the same topic group.

There are 80 long pairs, 64 in training and 16 in internal validation. Their inputs contain
1,071 to 1,857 pinned Liquid tokens. The longest complete training chat has 3,860 tokens;
the largest inference prompt plus the 2,048-token response allowance has 4,072 tokens. No pair
was truncated. All 219 exported targets passed the mechanical gates and the shared Sol rubric.
These are machine-reviewed synthetic targets. Independent human labels remain pending.

Luna authored new fictional documents. Long examples use protected, controlled input noise;
they do not represent the full range of natural ASR mistakes. The data still has repeated
count-check scenarios and limited language coverage. Earlier malformed or undersized source
attempts and the initial transport rehearsal remain as evidence and are excluded from this
release. The original 68 validation cases and their fixtures remain byte-for-byte unchanged.

`TRAINING_DATA.md` documents the exact formats, commands, checks, limitations, and next Colab
step. Training rows have four fields. The separate guide-format SFT export has only system,
user, and assistant. Supporting metadata and review critiques never enter supervised text.

The private training dataset is `jayshah5696/finetuning-labs-rewriter-training`, revision
`4610bc987246f266851b698c71bf250bd25a079f`. All 329 uploaded files were downloaded and
verified byte for byte. The receipt is `data/rewriter/releases/training-data.json`.
The dataset-check notebook has four code cells with empty outputs. Its three data-check cells
ran locally against the actual private Hub release, using cached authentication instead of
Colab Secrets. All 219 token spans and the 16-row selection passed. Actual Colab execution and
student training remain unverified.

## Luna Decisions audit, 27 October 2026

The new validator uses Arka's pinned `DecisionClient` with `openai/gpt-6-luna-decisions` through
OpenRouter. Each request asks five named criteria without a reference answer, source identity,
fixture expectation, or previous Sol verdict. Failure tests preceded implementation.

The 16-control probe and two full 375-item runs completed. The clarified questions accepted
71 of 80 authored acceptable outputs. Of 76 deliberately wrong outputs, they rejected 58,
left 17 uncertain, and incorrectly accepted one request changed into an assertion. The current
training-pilot audit has 157 pass, 48 uncertain, and 14 failure flags. Six flagged pairs have
unchanged input and output. Flags are disagreements, not confirmed data defects.

The 62 uncertain or flagged training pairs are in the second run's `review-queue.csv`.
`DECISION_VALIDATION.md` explains the source data, limited earlier Arka integration, current
decision client, criteria, thresholds, measured errors, and reproduction commands. These
thresholds are provisional. Independent human agreement has not been measured. The published
dataset, frozen 68 cases, and original Sol baseline records are unchanged.

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

Hugging Face and Modal authentication work. The user restored the OpenRouter credential in the
zshrc, and authenticated requests now work. The key was loaded only into process memory.
The shell configuration contents, symlink, and target were verified unchanged during evaluation.
No credential was written to Git, Hugging Face, run records, or another configuration file.

The source and scripts were committed as `ba6ec67`; the Hub card URI fix is `22cfabe`; the Colab
notebook and initial publication/run receipts are committed as `1b512a3`. No Git push was made.

The private dataset is `jayshah5696/finetuning-labs-rewriter`. Calibration revision
`10656b47271081e64764849857a68f66d4a552a5` contains 68 four-column rows. All 14 upload files were
downloaded and hash-verified. Runtime-assets revision `287fd2d44e6c8d9f044c348d0aa7267487be924a`
adds four verified files: the wheel, config, notebook, and asset manifest. The notebook's five
Python cells compile and have empty outputs; it has not been executed in Colab.

The Hub Dataset Viewer returned 501 because this private repository needs PRO or Enterprise
viewer access. Authenticated file downloads were verified. Repository visibility remains private.

## Liquid baseline result

Modal app `ap-V0ckxbaG1XjKTsMpmDkPtO`, call `fc-01M5YTGFB05E7KNGWJRQJHCXBM`, completed and
its artifacts were collected. The image used PyTorch 2.13.0+cu130 and Transformers 5.21.0 on an
NVIDIA L4, BF16, eager attention, greedy decoding, seed 42, and a 2,048-token generation limit.
The code commit was `ba6ec67884acb860a21b6be51f82b44f4393f06c`; the exact runner hash is saved
in the run receipt. The model and tokenizer revision are pinned in the config and raw records.

All 68 generations ended with EOS. None reached the output limit. Declared mechanical passes
were 53/68 overall: slack 8/12, transcribe 26/32, prompt 10/12, and polish 9/12. Long-case passes
were 0/8, and explicit identity preservation was 0/3. Added wrappers, omitted sections, and
summarized long inputs are visible in the raw outputs. The 53 mechanical passes remain pending
semantic assessment. There is no measured semantic quality score or Luna comparison.

The image used reference causal-convolution implementations because optimized causal_conv1d
was absent. No optimized speed claim is made. The 73 result/report files were uploaded and each
downloaded and verified at Hub revision `9d32281a7a2e8e4e66e1974e8f0240a1ffd71767`.

## Hosted baseline and matched judging

`openai/gpt-6-luna` generated all 68 cases through OpenRouter using Arka's pinned clean client
factory. The same `openai/gpt-6.1-sol` configuration assessed every Liquid and Luna candidate.
Both generation arms completed all 68 cases. All 136 judgments parsed and validated; none was
uncertain or an error. There were no output-limit truncations. No alternate model was used.

Sol-assessed whole-case passes were Liquid 2/68 and Luna 62/68. Equal-mode headlines were 4.2%
and 92.7%. Per-mode passes were slack 0/12 versus 12/12, transcribe 0/32 versus 28/32, prompt
1/12 versus 11/12, and polish 1/12 versus 11/12. Long-case passes were 0/8 versus 6/8. Explicit
identity preservation was 0/3 versus 3/3. Sol flagged critical errors in 46 versus three outputs.
These counts are model judgments, not independent human findings.

Paired outcomes were both pass 2, Luna-only pass 60, Liquid-only pass 0, neither verified pass 6.
All 68 scheduled cases stayed in each denominator. The report was recomputed from the exact
saved provider calls and matched both native metric files. Three new failure tests, written
before the comparison module, reject a changed judge, foreign candidate binding, and edited
derived summaries. Raw requests, responses, returned model IDs, usage, and critiques are saved.

The two setup calls are retained separately from the 204 benchmark API calls. OpenRouter
reported BYOK: its zero gateway charges are distinct from provider-reported upstream costs.
No zero-total-cost claim is made. GPU cost is outside those API usage fields.

All 217 new hosted/setup/report files were uploaded and downloaded for exact-byte verification
at Hub revision `f740b9d7908890f5317416c53ce27b174b55bdd9`. Publication stayed private. The six Luna
failures and a judge critical-error classification inconsistency are recorded for human review.
The independent locked test, human calibration, training data, and fine-tuning remain pending.
