# Rewriter training data

The 68 user-accepted cases are development validation. Their input, references, output fixtures,
and 64 source families stay outside training. `data/rewriter/validation-freeze.json` records their
identity. Keep the observed baseline failures. They help measure whether fine-tuning improves
the behavior that the user wants.

The training pilot uses new fictional documents. It has its own internal validation split.
Split whole topic groups before creating variants. Related modes and all variants stay together.
Use this internal split to select checkpoints. Use the fixed 68 cases for the before-and-after
comparison. An unseen final test remains future work.

The first retained release has 171 training pairs and 48 internal validation pairs from 55
documents. Eighty pairs are long, with inputs of 1,071 to 1,857 Liquid tokens. Of 320 planned
pairs, 101 are excluded. Both splits cover all four modes. See `release/quality.json` for the
counts by mode, length, and language.

The later [blind Luna Decisions audit](DECISION_VALIDATION.md) found 62 uncertain or flagged
pairs under provisional thresholds, plus a known false approval on the authored controls.
These results prioritize review. The published rows and the earlier Sol judgments remain
unchanged; neither evaluator establishes human-verified quality.

The verified private dataset is
[jayshah5696/finetuning-labs-rewriter-training](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training),
revision `4610bc987246f266851b698c71bf250bd25a079f`. All 329 uploaded files were downloaded
and compared byte for byte. The original validation repository was not changed.

## Source and quality checks

GPT-6 Luna authors canonical rewrites and imperfect inputs. Short and medium sources have three
teacher-authored variants plus one clean identity case. Long sources use a separate canonical
document followed by controlled surface errors. The latter teach formatting and light cleanup;
they do not cover the full range of natural speech-recognition mistakes.

The source planner has 80 documents across 20 topic groups and schedules four variants per
document. Sixteen topic groups belong to training and four to internal validation. Variants share
their target and are correlated. Count documents and topic groups as well as rows.

Every exported pair must pass these checks:

1. The source response parses correctly and has distinct variants. Protected names, quantities,
   quotes, and path literals survive in the input and target.
2. The input fits its length band. Long means 1,000 to 2,000 tokens with the pinned Liquid
   tokenizer. Complete training conversations fit 4,096 tokens. The inference prompt also leaves
   room for the same 2,048-token generation limit used by the baseline. Do not truncate a pair.
3. Sol passes meaning, terms and literals, mode, readability, and output discipline. Every
   criterion must pass, with no critical error. Missing, failed, invalid, or uncertain reviews
   cannot authorize a training target. Each review is bound to the exact input and target.
4. Exact, embedded, and near-copy checks find no overlap with the accepted validation cases,
   their output fixtures, or the local personal holdout. Topic groups and close wording cannot
   cross training and internal validation. Wording checks do not prove semantic independence.
5. The exported text matches the baseline prompt contract. Only assistant content through EOS
   receives training labels. System and user labels are -100. The native template's trailing
   newline after EOS is also masked. Reject empty or inconsistent assistant spans.

Sol uses the same rubric, model, and reasoning settings as the diagnostic baseline. Its judgments
are still not independently human calibrated. A release pass means that the automatic gates
passed. It does not establish perfect data quality.

## Files and commands

The release has four-column `train.jsonl` and `validation.jsonl` files. The `sft/` versions have
only `system`, `user`, and `assistant`, as in the supplied HTML guide. Source IDs, profile context,
literal rules, and review critiques remain in `supporting.json`. Only the profile or dictionary
context needed for the actual request enters the user text. Checks and review metadata do not.

The release is under `data/rewriter/training/v3-20261027/release/`. Read its `quality.json` for
actual accepted counts. `ledger.json` at the run root includes rejection reasons. `audit.json`
records the verification of exact saved calls, prompts, exports, and token spans.

Run the offline audit without an API credential:

```sh
uv run --locked --group data python -m finetuning_labs.rewriter.training \
  --run data/rewriter/training/v3-20261027 --audit
```

Generate a new run only with an explicit new path and an `OPENROUTER_API_KEY` in the environment:

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.training \
  --run artifacts/my-new-training-run --extended-long
```

The run saves each request before spending and each result before aggregation. Re-running the
same path reuses recorded calls. It never silently retries failed or interrupted calls. A changed
prompt or model setting requires a new run. `--reuse-from` can copy compatible earlier short and
medium calls without spending again. The audit verifies their exact requests and responses.

`notebooks/rewriter_dataset_check.ipynb` is the Colab entry point for data checks. It resolves
one immutable Hub revision, verifies file hashes, checks every native Liquid token span, and
selects 16 training rows for the first training proof. It loads no weights and does not train.
The notebook needs a read-only `HF_TOKEN` in Colab Secrets. All three data-check cells ran
locally against the actual private Hub release. The local run used cached Hugging Face auth
instead of Colab Secrets. Actual execution inside Colab remains unverified.

The quarantined transport rehearsal and earlier rejected long-source attempts are evidence.
They are excluded from the final training export. The first bulk attempt also preceded the final
medium-length gate and is not an eligible release under the stricter audit.

Publish only the verified release to a separate private repository:

```sh
uv run --locked --group data --group hosted python scripts/publish_training_data.py \
  --run data/rewriter/training/v3-20261027
```

The publisher uses an explicit file list, refuses a public or populated destination, checks for
configured credentials and selected personal content, then downloads every uploaded file at the
immutable Hub revision and compares its bytes. It leaves the validation repository unchanged.

## What to measure next

Start with a 16-row training proof on Colab L4. Check that the student can learn the exact prompt
format and response boundary before using the full pilot. Cover all four modes and include long
examples. Save the environment and checkpoint. Then fine-tune the 230M model with the full
training split and select a checkpoint using only internal validation.

Run that checkpoint on the unchanged 68 cases. Reuse the same Sol rubric and settings. Report
the pass rate for each mode and their equal-weight average, plus the raw pass count. Also report
long-case passes, exact identity preservation, literal violations, critical semantic errors,
truncations, and call failures. Training loss measures fit to targets; it is not rewrite quality.

Inspect failures and a varied sample of successful rewrites. Use those findings to add new
training source families. Keep the validation cases out of that expansion. The pilot still has
limited languages and repeated count-check scenarios. Natural ASR errors, dictionary corrections,
and more varied user-preferred edits need more coverage before a production-quality claim.
