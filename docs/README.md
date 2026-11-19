# Documentation Index

This directory contains project documentation, evaluation histories, data specifications, and historical notes for Finetuning Labs.

## Core guides and specifications

- [status.md](status.md): Chronological status log covering evaluations, dataset releases, and verification checks.
- [results-history.md](results-history.md): Summary of evaluation runs across the 141-case, 81-case, 84-case, and 68-case benchmarks.
- [training-data.md](training-data.md): Synthetic training data pipeline, Arka generation, Luna Decisions checks, Sol verification, and release reproduction.
- [validation.md](validation.md): Deterministic and semantic validation contract, metric definitions, and LLM-judge calibration rules.
- [decision-validation.md](decision-validation.md): Luna Decisions validator setup, criteria thresholds, and audit results.
- [colab-training.md](colab-training.md): Guide for running `notebooks/rewriter_train_l4.ipynb` on a Google Colab L4 GPU and uploading adapters to Hugging Face.
- [english-rewriter-experiment-log.html](english-rewriter-experiment-log.html): Standalone HTML experiment log covering setup, data iterations, Arka review, and model comparisons.

## Historical notes

- [repository-migration.md](repository-migration.md): Notes on extracting `finetuning-labs` from the `jadoo-labs` monorepo and [migration-commit-map.tsv](migration-commit-map.tsv).
- [data-expansion-workstreams.md](data-expansion-workstreams.md): Handoff contract for the six parallel data and training workstreams.
- [data-pilot-repair-20261030.md](data-pilot-repair-20261030.md): Results and coordinator verification from the October 2026 data pilot repair runs.

## Hub release note

`data/rewriter/PUBLIC_DATA_LICENSE.md` and `src/finetuning_labs/rewriter/hf_data.py` point to `https://github.com/jayshah5696/finetuning-labs`. The tracked receipt in `data/rewriter/releases/public-parquet.json` records the SHA-256 hashes of the currently uploaded Hugging Face files. The next `python -m finetuning_labs.rewriter.hf_data --publish` run will refresh `DATA_LICENSE.md` and `README.md` on the Hub and update `data/rewriter/releases/public-parquet.json`.

