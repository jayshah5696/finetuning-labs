# Contributing to Finetuning Labs

Thank you for your interest in contributing to Finetuning Labs.

## Setup

Install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). Then clone the repository and install the locked environment:

```sh
git clone https://github.com/jayshah5696/finetuning-labs.git
cd finetuning-labs
uv sync --locked --group data --group hosted
just check
just build
```

## Development workflow

- Write focused failure cases and unit tests before changing production code.
- Run `just check` (which runs pytest, Ruff lint, Ruff format check, and the benchmark audit) and `just build` before opening a pull request.
- Keep `notebooks/rewriter_train_l4.ipynb` synchronized with `src/finetuning_labs/rewriter/lora_training.py` via `scripts/build_lora_notebook.py`.

## Data and evaluation rules

- Never edit frozen validation or holdout files (`data/rewriter/validation-freeze.json`, `data/rewriter/holdout-freeze.json`, `data/rewriter/english-holdout-freeze.json`, or the splits they bind).
- Keep validation rows and their source families strictly excluded from training expansions.
- Keep benchmark rows limited to `input`, `output`, `mode`, and `checks`, and SFT rows limited to `system`, `user`, and `assistant`. Store provenance, literal rules, and review metadata in supporting files outside supervised text.
- Preserve completed judge verdicts, rejected candidates, and exact row hashes as evidence.
- Label model-judge results as diagnostic development-validation scores, not human gold labels.

## Credentials and privacy

- Never commit API keys, tokens, `.env` files, or personal transcription exports.
- Load credentials (`OPENROUTER_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`) into process memory from untracked local environment configuration or Colab Secrets.
- Public code contributions are licensed under Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
