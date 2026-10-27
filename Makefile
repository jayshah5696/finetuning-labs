.PHONY: sync test check build review export tokens

sync:
	uv sync --locked

test:
	uv run --locked pytest -q

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

# Evaluation and publication commands have explicit output paths and run IDs.
# See README.md. Neither is part of the default offline check.
