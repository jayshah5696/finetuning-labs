# Reproduce the comparison

Use the independent uv project at projects/finetuning-labs. Code and evaluation reports stay
in Git. This run does not upload code, outputs or reports to Hugging Face.

## Generate and collect

Choose a new run ID. Existing receipts prevent accidental submission of the same run twice.
The submit step checks all 84 SFT rows against the public frozen validation Parquet file.

```sh
uv run --locked --group data --group remote modal run --detach \
  scripts/modal_eval.py::submit_lora --run-id NEW_RUN_ID --gpu A10G
uv run --locked --group data --group remote modal run \
  scripts/modal_eval.py::collect --run-id NEW_RUN_ID
```

Use --gpu L4 when an L4 is available. Both arms run on the same requested device. The trained
adapter, base and dataset commits are pinned in submit_lora and copied into modal.json.
This run used app ap-hDqAPHepNc3tT01B1r0tpL and call fc-01M628A8KA2V4PYCTSTZS5C3FP.
The volume is finetuning-labs-evals. Its root for this run is lora-validation-20261028-a10g.

## Judge

Set OPENROUTER_API_KEY in the environment. This run loaded the existing authorized key from
the user's shell configuration into process memory. No key is stored in these records.
Use the unchanged configs/rewriter.toml judge settings and validation-bundle rubric.

```sh
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  --bundle data/rewriter/reviewed-v2/validation-bundle judge \
  --generation data/rewriter/evals/NEW_RUN_ID/stock \
  --output data/rewriter/evals/NEW_RUN_ID/stock-judge --workers 4
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  --bundle data/rewriter/reviewed-v2/validation-bundle judge \
  --generation data/rewriter/evals/NEW_RUN_ID/lora \
  --output data/rewriter/evals/NEW_RUN_ID/lora-judge --workers 4
```

Existing paid calls are read and verified rather than reissued. Failed or interrupted calls
remain visible. A changed request needs a new output directory. The separate setup control
is in the sibling lora-validation-20261028/setup directory and is outside the 168 assessments.

## Recompute scores and case reviews

This command needs no API key and makes no model calls. It verifies and reads the raw calls.

```sh
uv run --locked python -m finetuning_labs.rewriter.comparison \
  --bundle data/rewriter/reviewed-v2/validation-bundle \
  --run data/rewriter/evals/NEW_RUN_ID --left stock --right lora
make check
make build
```

It writes comparison.json, REPORT.md and case-reviews.csv in the comparison directory.
Physically missing raw call files abort the report; do not report a subset as an 84-case run.
All scheduled records were present and completed in this run. See runtime-audit.json for
the checks on identical prompts, configured decoding, versions, GPU and adapter identity.
