# Finetuning Labs

Finetuning Labs fine-tunes `LiquidAI/LFM2.5-230M` and `LiquidAI/LFM2.5-350M` with LoRA adapters as an English text rewriter for dictation cleanup across four modes (`/transcribe`, `/slack`, `/prompt`, and `/polish`). It builds reviewed synthetic data with the Arka pipeline (`openai/gpt-6-luna` generation, six Luna Decisions checks, and `openai/gpt-6.1-sol` verification), publishes Parquet datasets and PEFT LoRA adapters to Hugging Face, and evaluates stock and adapted models against deterministic rules and an LLM judge. See the [English rewriter experiment log](docs/english-rewriter-experiment-log.html) for the full narrative and setup.

## Current results

All scores below are LLM-judge diagnostics (`openai/gpt-6.1-sol`) on development validation, not independent human gold labels or results on an unseen final test. Liquid models use greedy decoding with a 2,048-token output budget on Modal A10 GPUs; hosted `openai/gpt-6-luna` uses provider-default sampling with reasoning off.

### Latest five-model English evaluation (141 cases, 1 November 2026)

Both rank-16 LoRA adapters (`230m` at `664f541`, `350m` at `b660a20`) were trained on the 1,979-row English training dataset (`36a75c009d008415fad557d49005ba4e5c73f4d7`) and evaluated on all 141 English validation cases (81 original frozen cases plus 60 expansion cases):

| Model | Arm | Strict passes (141) | Original 81 passes | Expansion 60 passes | Critical-error cases (141) | Truncated | Identity unchanged | Long passes (141) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LiquidAI/LFM2.5-230M` | base | 5/141 (3.5%) | 4/81 (4.9%) | 1/60 (1.7%) | 72 | 0 | 0/8 | 0/25 |
| `LiquidAI/LFM2.5-230M` | LoRA (`664f541`) | 64/141 (45.4%) | 53/81 (65.4%) | 11/60 (18.3%) | 60 | 1 | 7/8 | 4/25 |
| `LiquidAI/LFM2.5-350M` | base | 16/141 (11.3%) | 16/81 (19.8%) | 0/60 (0.0%) | 99 | 0 | 2/8 | 0/25 |
| `LiquidAI/LFM2.5-350M` | LoRA (`b660a20`) | 80/141 (56.7%) | 63/81 (77.8%) | 17/60 (28.3%) | 49 | 0 | 7/8 | 7/25 |
| `openai/gpt-6-luna` | cloud | 95/141 (67.4%) | 70/81 (86.4%) | 25/60 (41.7%) | 27 | 0 | 4/8 | 9/25 |

See the [141-case evaluation README](data/rewriter/evals/english-five-models-20261101/README.md) and [case-results.csv](data/rewriter/evals/english-five-models-20261101/case-results.csv).

### Earlier 81-case English comparison (29 October 2026)

On the 81 frozen English validation cases before the 60-case expansion (comparing the 1,864-row 350M LoRA against the earlier 1,500-row 230M LoRA at `ba71856`):

| Model | Arm | Passes (81) | Critical errors | Truncated | Identity unchanged | Long passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `LiquidAI/LFM2.5-230M` | stock | 4/81 (4.9%) | 33 | 0 | 0/8 | 0/13 |
| `LiquidAI/LFM2.5-230M` | LoRA (1,500-row) | 50/81 (61.7%) | 17 | 1 | 7/8 | 4/13 |
| `LiquidAI/LFM2.5-350M` | stock | 18/81 (22.2%) | 50 | 0 | 2/8 | 0/13 |
| `LiquidAI/LFM2.5-350M` | LoRA (1,864-row) | 62/81 (76.5%) | 13 | 0 | 7/8 | 7/13 |
| `openai/gpt-6-luna` | cloud | 73/81 (90.1%) | 1 | 0 | 4/8 | 10/13 |

See the [81-case report](data/rewriter/evals/english-all-models-20261029/REPORT.md), [case-reviews.csv](data/rewriter/evals/english-all-models-20261029/case-reviews.csv), and [docs/results-history.md](docs/results-history.md) for earlier 84-case and 68-case comparisons.

## Models and datasets on Hugging Face

- **350M LoRA adapter:** [jayshah5696/lfm2.5-350m-rewriter-lora](https://huggingface.co/jayshah5696/lfm2.5-350m-rewriter-lora) (base: [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M))
- **230M LoRA adapter:** [jayshah5696/lfm2.5-230m-rewriter-lora](https://huggingface.co/jayshah5696/lfm2.5-230m-rewriter-lora) (base: [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M))
- **Training and validation dataset:** [jayshah5696/finetuning-labs-rewriter-training](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training) (Parquet `default` and `sft` configurations; 1,979 train / 89 dev / 141 validation rows in the latest revision `36a75c0`, updated in place from the 1,864 / 89 / 81 release `00d59dd`)
- **Calibration benchmark dataset:** [jayshah5696/finetuning-labs-rewriter](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter)

```python
from datasets import load_dataset

data = load_dataset("jayshah5696/finetuning-labs-rewriter-training", "sft")
```

## Quickstart

Install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). Then clone the repository, sync locked dependencies, and run the offline test and audit suite:

```sh
git clone https://github.com/jayshah5696/finetuning-labs.git
cd finetuning-labs
uv sync --locked --group data --group hosted
just check
just build
```

Common `just` recipes include `sync`, `test`, `check`, `build`, `review`, `export`, `tokens`, `training-audit`, `reviewed-audit`, `notebook`, and `notebook-check`.

## Train in Colab

Open [notebooks/rewriter_train_l4.ipynb in Google Colab](https://colab.research.google.com/github/jayshah5696/finetuning-labs/blob/main/notebooks/rewriter_train_l4.ipynb) using an L4 GPU runtime. Because this repository is public, the Colab GitHub link works directly.

1. Add two secrets in the Colab Secrets panel and grant notebook access:
   - `HF_TOKEN`: Hugging Face write token for your destination model repository.
   - `WANDB_API_KEY`: Weights & Biases API key for loss logging.
2. In `CONFIG`, set `MODEL_SIZE` to `"350m"` or `"230m"`, and update `hub_model_id` if uploading under your own Hugging Face account. Public dataset and base-model downloads use `token=False`.
3. Run the cells in order. The notebook downloads the public SFT Parquet splits, trains a PEFT LoRA adapter with frozen base weights, selects the checkpoint by dev loss, evaluates validation loss once, and uploads the adapter weights, tokenizer files, and model card to Hugging Face.

See [COLAB_TRAINING.md](COLAB_TRAINING.md) for full configuration and CPU verification details.

## Evaluate

Offline benchmark audits and HTML/JSONL exports run without API keys:

```sh
just review
just export
just reviewed-audit
```

To run GPU inference on Modal and score outputs with the `openai/gpt-6.1-sol` judge, export `OPENROUTER_API_KEY` in your shell and see [scripts/README.md](scripts/README.md) and [docs/validation.md](docs/validation.md):

```sh
uv run --locked --group remote --group data --group hosted modal run --detach \
  scripts/modal_eval.py::submit_lora --run-id my-eval-run --gpu A10G \
  --model-size 350m --english-only --validation-source current
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/finetuning_labs/rewriter/` | Core package: benchmark audit, Arka generation pipeline, Luna Decisions validator, Sol judge runner, Hugging Face Parquet exporter, and LoRA training helpers. |
| `scripts/` | CLI scripts for building and checking the Colab notebook, running Modal GPU evaluations, judging outputs, and auditing dataset releases. |
| `notebooks/` | `rewriter_train_l4.ipynb`, the single self-contained Google Colab L4 LoRA training notebook. |
| `configs/` | TOML and JSON configurations for the rewriter benchmark, training data generation, model presets, and Luna Decisions validation. |
| `data/rewriter/benchmark-draft/` | Original 68-case calibration benchmark, 156 authored control fixtures, and supporting source snapshots. |
| `data/rewriter/current/` | Current reviewed source pool (`data/`, `sft/`, `validation-bundle/`, `supporting.json`, `quality.json`, `reviews/`, and `evidence/`). |
| `data/rewriter/reviewed-v2/` | Historical reviewed-v2 release (65 train, 17 dev, 84 validation rows) and its validation bundle. |
| `data/rewriter/training/` | Historical synthetic training pilots (`pilot-20261027`, `v1-20261027` through `v3-20261027`). |
| `data/rewriter/evals/` | Saved evaluation runs, raw generation and judge records, CSV comparisons, and reports across stock, LoRA, and Luna arms. |
| `data/rewriter/releases/` | Verification and publication receipts for Hugging Face exports, tokenizer checks, and notebook CPU integration tests. |
| `data/rewriter/reviews/` | Saved LLM review reports and distribution analyses. |
| `data/rewriter/decision-audits/` | Saved Luna Decisions audit runs and review queues. |
| `data/rewriter/expansions/` | Staged data expansion workstream outputs and quarantine manifests. |
| `docs/` | Project documentation, experiment log HTML, evaluation history, and historical notes. |
| `tests/` | Pytest unit and integration test suite. |

## Data and licensing

This repository uses an explicit split between code licensing and data licensing:

- **Code and documentation:** Licensed under the [Apache License, Version 2.0](LICENSE) (Copyright 2026 Jay Shah; see [NOTICE](NOTICE)).
- **Data under `data/`:** Marked `other` as documented in [data/rewriter/PUBLIC_DATA_LICENSE.md](data/rewriter/PUBLIC_DATA_LICENSE.md). Do not assume an Apache-2.0 grant for the dataset. Specifically, 30 original validation cases derive from GPL-3.0 TypeWhisper Linux fixtures (attribution and license text are preserved under `data/rewriter/benchmark-draft/supporting/TYPEWHISPER-LICENSE.txt`). The remaining benchmark cases and all synthetic training and development pairs were authored for this project. No personal transcription history or audio recordings are included.

## Documentation

- [Documentation index](docs/README.md)
- [English rewriter experiment log (HTML)](docs/english-rewriter-experiment-log.html)
- [Project status log](docs/status.md) and [Results history](docs/results-history.md)
- [Training data specification](docs/training-data.md)
- [Validation contract and metrics](docs/validation.md)
- [Luna Decisions validation](docs/decision-validation.md)
- [Colab L4 training guide](COLAB_TRAINING.md)
- [Contributor guide](CONTRIBUTING.md), [Agent rules](AGENTS.md), and [Security policy](SECURITY.md)
