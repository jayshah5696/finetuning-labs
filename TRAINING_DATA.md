# Rewriter training data

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

`scripts/curate_arka_v2.py` records the exact parent corrections. The release preparer consumes
saved attempts and reviews without making model calls. Use a fresh output path:

```sh
uv run --locked --group data --group hosted python scripts/prepare_reviewed_release.py \
  --pool artifacts/arka-curated-v2-round3 --output artifacts/reviewed-v2-rebuilt
```

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

The publisher adds `reviewed-v2/` to the existing private training repository and redirects its
default data configs there. It refuses to overwrite that version or publish to a public repo.
It scans for credentials and selected personal content. It downloads every new file at the
immutable upload revision and compares its bytes. Historical data files remain unchanged.

The private [reviewed v2 publication](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/tree/70715437e3d6b67ef47eb89c83ccb1a66038082e/reviewed-v2)
is revision `70715437e3d6b67ef47eb89c83ccb1a66038082e`. All 69 uploaded files were downloaded
and compared byte for byte. The receipt is `data/rewriter/releases/reviewed-v2.json`.

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.reviewed_publication \
  --release data/rewriter/reviewed-v2
```

## Colab and the next measurement

Use [notebooks/rewriter_reviewed_data_check.ipynb](notebooks/rewriter_reviewed_data_check.ipynb).
Add a read-only `HF_TOKEN` in Colab Secrets. The notebook resolves one immutable Hub revision,
verifies the manifest, checks all 166 exact reviews and token spans, and prepares 16 training
examples across the four modes. It loads no weights and runs no training. The earlier
`rewriter_dataset_check.ipynb` is retained as a retired v1 artifact.

All three data cells ran locally against the actual private Hub revision, with cached Hub
authentication replacing Colab Secrets. `scripts/check_reviewed_colab.py` reproduces that check.
The receipt is `data/rewriter/releases/colab-reviewed-data-check-local.json`. Actual execution
inside Colab and L4 training remain unverified.

The next step is a 16-row memorization proof on Colab L4. Check assistant-only training loss,
EOS handling and the ability to learn the prompt format. Then train on the 65 training pairs
and choose a checkpoint using only the 17 development pairs. Memorization and low loss do not
measure generalization.

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
