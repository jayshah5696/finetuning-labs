# Preparation and evaluation commands

Run these from the Finetuning Labs project directory. Use `uv` and the locked dependency groups.

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

```sh
uv run --locked python scripts/report_generation.py \
  --generation data/rewriter/evals/baseline-20261027/liquid \
  --output data/rewriter/evals/baseline-20261027/liquid-mechanical
```
