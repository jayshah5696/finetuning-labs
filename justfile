default:
    @just --list

sync:
    uv sync --locked

test:
    uv run --locked --group data --group hosted pytest -q

check: test
    uv run --locked ruff check .
    uv run --locked ruff format --check .
    uv run --locked finetuning-labs audit

build:
    uv build

review:
    uv run --locked finetuning-labs review --output artifacts/benchmark-review.html

export:
    uv run --locked finetuning-labs export --output artifacts/benchmark-guide.jsonl

tokens:
    uv run --locked --group data finetuning-labs tokens --report artifacts/benchmark-tokens.json

training-audit:
    uv run --locked --group data python -m finetuning_labs.rewriter.training --run data/rewriter/training/v3-20261027 --audit

# The target above audits historical mechanics; that pilot failed independent data review.
reviewed-audit:
    uv run --locked --group data --group hosted python -c 'from pathlib import Path; from finetuning_labs.rewriter.reviewed_publication import audit_publication; print({"verified_files": len(audit_publication(Path("data/rewriter/current")))})'
    uv run --locked finetuning-labs --bundle data/rewriter/reviewed-v2/validation-bundle audit
    uv run --locked finetuning-labs --bundle data/rewriter/current/validation-bundle audit

notebook dataset_revision="36a75c009d008415fad557d49005ba4e5c73f4d7" model_size="350m":
    uv run --locked python scripts/build_lora_notebook.py --dataset-revision {{dataset_revision}} --model-size {{model_size}}

notebook-check output="artifacts/lora-notebook-check":
    uv run --locked --group data --group training python scripts/check_lora_notebook.py --output {{output}}

# Evaluation and publication commands have explicit output paths and run IDs.
# See README.md. Neither is part of the default offline check.
