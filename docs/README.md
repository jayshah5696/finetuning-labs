# Documentation Index

This directory contains project documentation, evaluation histories, data specifications, and historical notes for Finetuning Labs.

## Core guides and specifications

- [status.md](status.md): Chronological status log covering evaluations, dataset releases, and verification checks.
- [results-history.md](results-history.md): Summary of evaluation runs across the 141-case, 81-case, 84-case, and 68-case benchmarks.
- [training-data.md](training-data.md): Synthetic training data pipeline, Arka generation, Luna Decisions checks, Sol verification, and release reproduction.
- [validation.md](validation.md): Deterministic and semantic validation contract, metric definitions, and LLM-judge calibration rules.
- [decision-validation.md](decision-validation.md): Luna Decisions validator setup, criteria thresholds, and audit results.
- [COLAB_TRAINING.md](../COLAB_TRAINING.md) (also indexed at [colab-training.md](colab-training.md)): Guide for running `notebooks/rewriter_train_l4.ipynb` on a Google Colab L4 GPU and uploading adapters to Hugging Face.
- [english-rewriter-experiment-log.html](english-rewriter-experiment-log.html): Standalone HTML experiment log covering setup, data iterations, Arka review, and model comparisons.

## Historical notes

- [repository-migration.md](repository-migration.md): Notes on extracting `finetuning-labs` from the `jadoo-labs` monorepo and [migration-commit-map.tsv](migration-commit-map.tsv).
- [data-expansion-workstreams.md](data-expansion-workstreams.md): Handoff contract for the six parallel data and training workstreams.
- [data-pilot-repair-20261030.md](data-pilot-repair-20261030.md): Results and coordinator verification from the October 2026 data pilot repair runs.
