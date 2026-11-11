.PHONY: sync test check build review export tokens training-audit reviewed-audit

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

# Evaluation and publication commands have explicit output paths and run IDs.
# See README.md. Neither is part of the default offline check.
