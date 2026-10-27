# Preparation and evaluation commands

Run these from the Finetuning Labs project directory. Use `uv` and the locked dependency groups.

Current data commands are in [TRAINING_DATA.md](../TRAINING_DATA.md).
`curate_arka_v2.py` preserves exact corrections after data review. `prepare_reviewed_release.py`
exports only exact Opus passes and packages explicit evidence. `recover_opus_report.py` restores
the omitted third-round review array from the reviewer's saved command without inventing labels.
The `reviewed_publication` module verifies and uploads the additive private `reviewed-v2/`
version. `publish_training_data.py` belongs to the retired pilot and must not be used for v2.
`check_reviewed_colab.py` runs the three actual notebook data cells locally against the private
Hub release, with cached Hub authentication in place of Colab Secrets. It does not run training.

`append_long_cases.py` preserves the historical authoring script used to add the eight long
examples. It contains the four fictional source documents and the controlled edits and attacks.
It expects the original 60-case bundle and prints an `apply_patch` request. It does not alter
files itself. The committed 68-case bundle is the canonical result. Do not rerun the append on it.

The `finetuning-labs` command audits, renders, exports, and measures token lengths. These commands
are in the Makefile. They do not make inference calls.

The reusable evaluation module keeps exact generation requests, raw responses, model identity,
usage, prompt hashes, context limits, and failure status. Existing saved calls are never silently
reissued. Use a new run directory for changed settings. API errors contain only safe error types
and HTTP statuses. Personal transcription bundles are refused by the shared CLI.

```sh
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  generate --model luna --workers 4 --output data/rewriter/evals/RUN/luna
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  judge --generation data/rewriter/evals/RUN/luna \
  --output data/rewriter/evals/RUN/luna-judge --workers 4
```

Hosted calls use `OPENROUTER_API_KEY`, the requested `openai/gpt-6-luna` and
`openai/gpt-6.1-sol` models, and Arka's shared client factory at the pinned clean commit. They
use the OpenRouter Responses API. The decision-model API is not substituted for text generation.
All Sol scores remain diagnostic until checked against human judgments.

`modal_eval.py` runs the same Liquid generation function on an L4. It uses a persistent Modal
volume, a detached submission, and a saved function-call receipt. It uploads only package code
and the shared calibration bundle. It never mounts the personal directory.

```sh
uv run --locked --group remote modal run --detach scripts/modal_eval.py::submit --run-id RUN
uv run --locked --group remote modal run scripts/modal_eval.py::collect --run-id RUN
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  judge --generation data/rewriter/evals/RUN/liquid \
  --output data/rewriter/evals/RUN/liquid-judge --workers 4
```

The Hub packager copies an explicit file allowlist. It keeps the four-column dataset separate
from supporting records and preserves the TypeWhisper GPL license. A changed package or an
existing public destination is rejected. After uploading, it downloads every file at the new
immutable Hub revision and checks its hash.

```sh
uv run --locked --group data python -m finetuning_labs.rewriter.hub package \
  --output artifacts/huggingface --code-commit COMMIT
uv run --locked --group data python -m finetuning_labs.rewriter.hub upload \
  --package artifacts/huggingface --repo-id jayshah5696/finetuning-labs-rewriter \
  --receipt data/rewriter/releases/huggingface.json
```

`publish_runtime_assets.py` uploads exactly the wheel, config, and Colab notebook to the private
dataset and verifies the resulting bytes. The notebook resolves one immutable Hub revision at
startup and records asset hashes. Its code cells are syntax-checked; Colab execution is pending.

`report_generation.py` records mechanical findings for the collected stock Liquid L4 run.
It deliberately leaves the semantic quality score empty. `publish_liquid_results.py` uploads
only the 68 known case records, run metadata, receipt, and mechanical report, then verifies every
published file. The original raw candidates remain unchanged.

The comparison command recomputes scores from bound provider records. It rejects a changed
judge/rubric or a judgment attached to a different candidate, and ignores edited derived scores.
Both models keep the same behavioral prompt and the same 68 scheduled inputs.

```sh
uv run --locked python -m finetuning_labs.rewriter.comparison \
  --run data/rewriter/evals/baseline-20261027
```

`publish_hosted_results.py` uploads an explicit list of Luna generations, both Sol judgment
sets, setup calls, and the paired report. It verifies all bytes at an immutable private Hub
revision. Existing published results are refused rather than silently replaced.

```sh
uv run --locked python scripts/report_generation.py \
  --generation data/rewriter/evals/baseline-20261027/liquid \
  --output data/rewriter/evals/baseline-20261027/liquid-mechanical
```

## Training data

`python -m finetuning_labs.rewriter.training` prepares separate synthetic training pairs.
`--extended-long` authors long canonical documents before adding controlled noise. `--audit`
checks an existing run offline. See [TRAINING_DATA.md](../TRAINING_DATA.md) for exact commands.

`publish_training_data.py --run data/rewriter/training/v3-20261027` publishes the verified
training release to its own private Hugging Face repository and verifies every uploaded file.
It excludes the validation bundle, the quarantined rehearsal, and personal transcripts.

## Blind decision validation

`python -m finetuning_labs.rewriter.decision_validation --run NEW_PATH --scope all` uses Arka's
native decision client to evaluate five named criteria without a reference or prior verdict.
It checks the 156 authored controls and 219 training-pilot pairs, saves exact calls, and writes
a separate review queue. It changes no training rows. Use the locked `hosted` dependency group
and an `OPENROUTER_API_KEY`. See [DECISION_VALIDATION.md](../DECISION_VALIDATION.md) for results
and the limits of the provisional probability thresholds.
