# Rewriter training data

## Current English export, 1,864 training examples

The public dataset now selects English only. It has 1,864 train, 89 dev and 81 validation
pairs. The English dev and validation cases keep their original text and order, bound by
`data/rewriter/english-holdout-freeze.json`. The complete multilingual 100/84 source holdouts
remain byte-identical in Git, alongside all earlier training examples. The source training
pool has 2,000 rows. Filtering the public view does not rewrite those sources.

The latest 500 fictional English episodes target failures in the uploaded 1,500-example
adapter. They include recipient and amount exclusions, explicit spoken corrections with
surrounding context, meaningful transcription words, deliberately unfinished thoughts,
prohibitions, conditional alternatives, supplied code and complete long sections.
New cases are independent analogues rather than copies of validation inputs.

| New examples | Short | Medium | Long | Total |
| --- | ---: | ---: | ---: | ---: |
| /slack | 20 | 20 | 10 | 50 |
| /transcribe | 100 | 80 | 20 | 200 |
| /prompt | 50 | 60 | 40 | 150 |
| /polish | 30 | 40 | 30 | 100 |

There are 125 exact identity pairs across the new set. Long inputs contain at least 1,000
tokens under the pinned Liquid tokenizer. Fresh reverse-direction episodes balance which
person, amount or corrected location is intended. Existing reviewed pairs are never edited
to create these reversals. The exact final selection and balance report are bound by row hashes.

The additions have 500 distinct targets across 52 declared domains. Of 375 nonidentity inputs,
368 start lowercase. All 100 long targets have paragraph breaks, 70 have numbered sections
and 93 contain quoted text. Their input lengths are 1,003 to 1,558 tokens. The 33 supplied-code
pairs pass exact-block and added-code checks. These measurements describe coverage, not meaning.

The native Arka pipeline uses the pinned latest clean commit, GPT-6 Luna generation and six
Luna Decisions checks. The admission policy below remains unchanged. Each retained exact pair
has either one strong native result and a Sol pass, or two separate Sol passes for a complete,
non-vetoed native result. Rejected and uncertain Sol verdicts cannot be promoted by retrying.
Plans, raw Sol calls and reviews are in `current/evidence/english-critical/`. Previous boost
evidence remains intact. These are synthetic data and model judgments. They do not establish
how well the inputs represent recordings or replace human quality labels.

The final selection has 58 native-strong pairs with one Sol pass and 442 pairs with two separate
Sol passes, for 942 saved exact-row verification calls. Direction counts are 19 larger versus
24 smaller intended amounts, 14 A versus 16 B final locations, and 27 reverse versus 31 forward
recipient cases. The full English training view has 407 long inputs and 361 exact identity pairs.
See the measured reports in `data/rewriter/releases/english-*-distribution.json` and
`english-expansion-check.json`.

The final Sol audit read all 33 supplied-code pairs and all 49 nonidentity correction and
unfinished transcription pairs. It found two contradictory explanation requests and one
unapplied spoken correction that the earlier per-row passes missed. Those three pairs are
excluded. Replacements retain their own native decisions and exact Sol passes, and also
passed the targeted audit. Original verdicts and both selections remain in the evidence.
This is a known limit of automatic judging, not a reason to relabel the original passes.
The findings are in `data/rewriter/reviews/sol-english-quality-findings-round1.json`.

Use the same Colab notebook, with `MODEL_SIZE` set to `350m` or `230m`. The user runs training.
The 350M adapter uses a different model repository because it requires a different base.
After upload, compare both models on the same frozen 81 English cases with the same Sol
rubric. The current available 230M LoRA passes 50/81, with 17 critical-error outputs among 80
reviewed outputs and one truncation. Its earlier 51/81 differs by one conflicting saved judgment.
The new 350M LoRA passes 62/81 and hosted Luna passes 73/81 on the same full English validation set.
The 230M upload still records the older 1,500-row training data. Neither lower dev loss nor a
larger model proves improvement by itself.

The recorded preparation commands below ran against the original 1,500-row source snapshot.
They produce new stochastic examples, not a byte-identical regeneration. Resume uses saved
plans and completed responses. A supplemental source pool adds fresh long requests; it never
relabels failed pairs. Do not rerun authoring against the already appended current release.
Use the audit commands below to verify the existing data without new model calls.

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.english_critical \
  --run artifacts/english-critical
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.english_critical \
  --run artifacts/english-critical --complete
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.english_critical \
  --run artifacts/english-critical --complete --complete-pool fidelity-long
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.english_critical \
  --run artifacts/english-critical --balance
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.english_critical \
  --run artifacts/english-critical --apply
```

Run completion only when the main selection is short. The final append requires all 500
verified pairs, the exact identity quota and a valid direction-balance report. Use
`make reviewed-audit` to verify committed data and review evidence without new model calls.

```sh
uv run --locked --group data --group hosted python scripts/check_english_release.py --tokenizers
```

## Previous 1,500-example source pool

The previous source pool, retained inside `data/rewriter/current/`, had 1,500 train, 100 dev
and 84 validation pairs. The original 1,000 training pairs are an unchanged prefix. Dev and
validation data and SFT files retain their exact original bytes.

Its 500 added fictional episodes targeted failure categories observed in the original adapter:
changed main actions, mishandled spoken corrections, missing conditional branches, lost supplied
code or quotes, changed actors or hedges, and missing final sections in long documents.
They do not copy the validation cases. Training has 300 Slack, 450 transcription, 400 prompt
and 350 polish pairs. The additions include 180 long inputs, each with 1,000 to 2,000 tokens
under the pinned Liquid tokenizer. Long examples include paragraphs, numbered sections and quotes.

Generation uses the pinned latest clean Arka pipeline with GPT-6 Luna. Luna Decisions checks
candidate input, output quality, facts, intent, literals and mode. Exact literals, identity,
token limits, transcription words where appropriate, and overlap receive mechanical checks.
All six native decision results must exist. A score at most 0.1 with confidence at least 0.8
blocks the pair. A native final export with all scores at least 0.8 and confidence at least 0.2
requires one passing Sol review. Other complete, non-vetoed native results require two separate
passing Sol calls on the exact same pair. These probabilities are not calibrated quality scores.

Each GPT-6.1 Sol call grades the input, output and pair separately, with exact row hashes.
The two calls use the same model and rubric; they are not independent human labels.
Failed and uncertain Sol verdicts remain excluded. Only incomplete or malformed calls may retry,
with the original attempt preserved. Raw calls, rejected candidates, plans, configs, Arka scores
and exact review reports stay in Git under `current/evidence/boost/` and `current/reviews/`.
Only Parquet data, the dataset card and license notices are published to Hugging Face.

`quality.json` records measured counts and separates the original selection statistics from the
new expansion. `holdout-freeze.json` protects dev and validation. The public publication receipt
pins the notebook to the same data commit and expected counts. The final notebook check encodes
every row, then exercises real LoRA training, save and reload on a tiny CPU model. GPU training
and the new adapter's semantic comparison are still required after the user reruns Colab.

## Original 1,000-example selection

This section records the original selection, now retained inside `data/rewriter/current/`.
It had exactly 1,000 training pairs and 100 development pairs. Each mode had 250 train and 25 dev pairs.
The 84 validation data and SFT rows are byte-identical to reviewed v2. Source groups do not cross
splits. Training has 1,000 distinct targets and source groups. Dev has 100 of each.

The [public Hugging Face dataset](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/viewer/default/train)
uses standard Parquet splits. The default viewer shows input, output, mode and checks. Select
`sft` for the guide's system, user and assistant columns. Code, notebooks and verification
records stay in Git. Dataset publication no longer uploads those artifacts to Hugging Face.

This release retains 65 train, 17 dev and all 84 validation pairs from v2. It adds 935 train and
83 dev pairs from new fictional scenarios. All retained exact inputs, outputs and pairs have
completed Opus data passes. The user later selected GPT-6.1 Sol for the remaining implementation
review and fix loop. Sol approved the exact notebook and runtime script. No further Opus calls
were made after that change. The code review and data verdicts are separate evidence.

The native Arka generation and six-dimension Luna Decisions pipeline generated 1,720 new
candidates across the main run and two top-ups. Schema, length or context checks rejected 57.
One additional row failed a mechanical overlap check. Opus completed 1,648 exact data reviews:
1,330 pass, 241 fail and 77 uncertain. Fourteen rows lack a completed review and are excluded.
Completed failures and uncertain verdicts were preserved. Only interrupted or malformed review
calls received separate attempts. The final quotas leave 312 other passing candidates unused.

Training has 139 long inputs, dev has 14 and validation has 13. Long inputs contain 1,004 to
1,906 pinned Liquid tokens. All 139 long training targets have paragraph breaks, 92 have numbered
sections and 116 have quoted text. The maximum complete chat is 4,029 tokens. The largest prompt
plus the 2,048-token generation reserve is 4,091. Nothing was truncated.

The selected new training rows cover 68 declared domains. Training contains 188 unchanged pairs.
Of 715 nonidentity English inputs, 666 start lowercase. The most frequent eight-word target
sequence appears in five of the 1,000 training documents. These are distribution checks, not
semantic quality scores. The exact report binds the training file hash in
`data/rewriter/reviews/scaled-v3-distribution.json`. Recompute it with:

```sh
uv run --locked python scripts/data_distribution.py \
  --release data/rewriter/current \
  --output artifacts/recomputed-current-distribution.json
```

Data rows still contain only input, output, mode and checks. SFT rows contain only system, user
and assistant. Required language, dictionary and profile context enter the user request.
Source IDs, probabilities, critiques and other metadata never enter supervised text.
Every selected row retains all six native Decisions scores. Those scores prioritize review;
the independent exact-row Opus verdict adjudicates disagreements. Mechanical failures stay blocked.

The original release builder and artifact packagers are retained in Git history at commit
`91e36a81df60f2520fc9334ef06ec27d729a15ad`. They selected the reviewed rows from the preserved
Arka run evidence. The current data auditor verifies the original row hashes, exact reviews,
source groups and frozen validation bytes without requiring those retired code bundles.
The source plans, pipeline/rubric configs, saved reviews and exclusion records remain in Git.
The current data-only publication receipt is `data/rewriter/releases/public-parquet.json`.
The former artifact-bundle receipt is historical. The dataset-only publisher removes non-data
files from the current Hub trees; previous commits remain in history.

The notebook loads the public SFT data and pinned 230M base, trains LoRA using dev loss for
checkpoint selection, records loss in W&B and uploads the final adapter to a model repository.
Add HF_TOKEN with write access and WANDB_API_KEY in Colab Secrets.
[COLAB_TRAINING.md](COLAB_TRAINING.md) explains this workflow and its limits.
The actual notebook cells passed a tiny native LFM CPU check with all public data encoded,
frozen base weights, adapter updates and exact adapter reload. W&B was offline and the model
upload was mocked. Actual 230M L4 training remains pending. Human labels, human judge calibration
and an unseen final test also remain pending.

## Historical reviewed v2 release

The remaining sections describe the previous release and its reproduction. Their private Hub
layout and data-check-only notebooks are superseded. Use the current L4 LoRA notebook above.

Use the reviewed v2 release in `data/rewriter/reviewed-v2/`. It contains 65 training pairs,
17 development pairs and 84 validation pairs. Every released input, output and pair has an
Opus 5.5 pass bound to its exact contents. These are model judgments, not human gold labels.

The original 219-pair pilot failed the first independent Opus review. Do not train on it.
Its files and publication receipts remain for reproduction. The rejected pilot's repeated
count-check template, artificial noise and padded long texts are documented in
[data/rewriter/reviews/opus-first-pass.md](data/rewriter/reviews/opus-first-pass.md).

## Sources and splits

GPT-6 Luna generated one fictional pair per scenario through clean Arka commit
`5d353740ae1dd34742841071752814028db1489a`. The scenario briefs cover different subjects and
genres. Long Slack examples are pasted threads. Long transcription inputs use missing punctuation,
restarts and spoken corrections. Prompt targets remain requests. Polish targets improve prose.

The final release has 98 new synthetic pairs and all 68 original authored benchmark cases.
Training has 65 independent source groups and 65 distinct targets. Development has 17 of each.
Validation has 84 rows from 80 groups and 80 distinct targets. The related long benchmark variants
share source groups. No group crosses any of the three splits.

Use `train` for fitting. Use `dev` for checkpoint selection. Use `validation` for the fixed
before-and-after comparison. It is development validation, not an unseen final test. The
validation split includes 16 new pairs and two corrected versions of original cases:

- `tw-profiles-03` uses consistent punctuation in its numbered list.
- `polish-06` states the intended timeout explicitly instead of forcing an ambiguous reading.

The original 68-case files, references and historical baseline scores remain byte-for-byte
unchanged. Rerun the stock Liquid baseline on v2 before comparing a fine-tune with v2 scores.

Thirty validation cases derive from TypeWhisper authored fixtures. The source snapshot and
GPL v3 notice are preserved. Training and development contain new synthetic scenarios. No
personal history text or recordings were sent to the models or included in this release.
The ten selected personal examples remain local, pending preferred corrections.

## How quality was verified

Arka ran its native `prompt_based_generator`, normalization, exact and length checks,
`decision_panel`, and `labeling_engine`. The panel uses `openai/gpt-6-luna-decisions`
through OpenRouter's alpha endpoint. It tests six dimensions:

1. The input is plausible, clear enough to recover, and free of padding or authoring rules.
2. The output is readable and suitable for the requested mode.
3. The pair preserves facts, conditions, negation and uncertainty.
4. Questions, requests, proposals and explicit corrections retain their intent.
5. Names, quantities, quoted text and code literals survive permitted edits.
6. The rewrite follows the mode and adds no wrapper or formatter commentary.

The final native check scored all 168 candidate pairs. Arka automatically accepted 75.
Its weighted-score and confidence policy left the others for review. All six scores and the
automatic policy outcome remain in the supporting records. These scores are not measured
probabilities of agreement with people.

Opus reviewed data only. It did not generate examples, edit data, review code, or invoke other
models. Its packets contain mode instructions, required context, input and output. They omit
Luna scores, previous verdicts and separate reference answers. The first full replacement review
covered 161 pairs. The parent corrected 23 rows and added seven complete examples. Opus then
rechecked all 30 changed or new pairs. The release excludes:

- `arka-v2-prompt-08`, whose output still drops an explicit budget constraint.
- `arka-v2-transcribe-23`, whose self-correction cue remains ambiguous.

All 166 retained rows pass input, output and pair verification. Mechanical failures cannot be
overridden by an Opus pass. Each verdict binds to the exact row hash. Changing a row invalidates
its old review. The incomplete third-round train/dev report and its mechanically recovered
array are both retained. The recovery uses the reviewer's original saved verdicts and critiques.

The export also checks literal rules, exact and close wording overlap with the original benchmark,
its fixtures and the local personal holdout, and all source-group boundaries. Close wording checks
do not prove semantic independence. It verifies the actual pinned Liquid chat template, assistant
content through EOS, and masked system/user tokens. No example is truncated.

There are 33 long inputs of 1,094 to 1,906 Liquid tokens. The maximum complete chat has 4,029 tokens.
The largest prompt plus the 2,048-token generation allowance has 4,091 tokens. Use a 4,096-token
context. There are 19 short, two medium and two long unchanged pairs.

## Files and reproduction

Main rows have only `input`, `output`, `mode` and `checks`. Guide-format SFT rows have only
`system`, `user` and `assistant`. Only needed language, profile or dictionary context enters
the user request. Source records, checks and review labels never enter supervised text.

`supporting.json` contains source identity, six native decisions, Opus verdicts and token counts.
`quality.json` contains coverage and limits. `excluded.json` records the two unresolved rows.
`reviews/` and `evidence/` preserve the attempts, corrections, exact reviews and native records.
`validation-bundle/` works with the existing evaluation runner. Its judge rubric matches v1.
It has no new authored output-control fixtures.

Run the offline project checks:

```sh
make check
make build
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.reviewed_publication \
  --release data/rewriter/reviewed-v2 --prepare-only
uv run --locked finetuning-labs --bundle data/rewriter/reviewed-v2/validation-bundle audit
```

The authoring entry point creates a new stochastic run. It does not recreate the same examples.
It requires `OPENROUTER_API_KEY` in the environment. Do not reuse a completed run path with
changed prompts or scenarios.

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.arka_pipeline \
  --run artifacts/new-arka-run
```

`scripts/curate_arka_v2.py` records the exact parent corrections. Historical bundle builders
are available in Git history. The current tree contains the single supported training notebook.

To reproduce the six data and SFT files from committed evidence alone:

```sh
uv run --locked --group data python -m finetuning_labs.rewriter.reviewed_release \
  --pool data/rewriter/reviewed-v2/evidence/pool \
  --review data/rewriter/reviewed-v2/reviews/round2-train-dev.json \
  --review data/rewriter/reviewed-v2/reviews/round2-validation.json \
  --review data/rewriter/reviewed-v2/reviews/round3-train-dev-recovered.json \
  --review data/rewriter/reviewed-v2/reviews/round3-validation.json \
  --output artifacts/reviewed-v2-from-evidence
```

The former reviewed-v2 publisher is retired for writes. The original publication revision was
`70715437e3d6b67ef47eb89c83ccb1a66038082e`. Its 69 files were downloaded and compared byte for
byte at the time. The historical receipt is `data/rewriter/releases/reviewed-v2.json`.
Publish only Parquet data, cards and source-license notices with:

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.hf_data --publish
```

## Historical v2 data check and comparison contract

The superseded `rewriter_reviewed_data_check.ipynb`
used a read-only HF_TOKEN and one immutable Hub revision. It
verified the manifest, checked all 166 exact reviews and token spans, and prepared 16 training
examples across the four modes. It loaded no weights and ran no training. The earlier
`rewriter_dataset_check.ipynb` is also retired. Both are available only in Git history.

All three data cells ran locally against the actual private Hub revision, with cached Hub
authentication replacing Colab Secrets. Its checker is also retained in Git history.
The receipt is `data/rewriter/releases/colab-reviewed-data-check-local.json`. Actual execution
inside Colab and L4 training remain unverified.

The earlier plan proposed a 16-row memorization proof followed by full training on 65 pairs.
The current notebook instead trains LoRA on the 1,000-row release. Memorization and low loss
do not measure generalization.

Run stock and tuned Liquid on the same 84 v2 validation cases with the same decoding settings.
Keep the same Sol output rubric and judge settings in both arms. Report the verified pass rate
in each mode and the average of the four mode rates with equal weights. Also report the pooled
pass count, long-case passes, exact unchanged-input preservation, literal violations, critical
meaning errors, truncations and failed calls. Missing, uncertain and failed outputs stay in
the denominator and do not count as passes.

This remains a small reviewed pilot. English accounts for 160 of 166 rows. Only six rows are
German or Portuguese. Dictionary and profile coverage remain narrow, uncertainty-preservation
cases are common, and no recordings establish that the synthetic noise matches real ASR.
Independent human labels, an unseen final test and student improvement are still unmeasured.
