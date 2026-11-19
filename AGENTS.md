# Agent and Contributor Guidelines

Use `uv` and [just](https://github.com/casey/just) (`justfile`). Keep this project independent of any parent uv workspace.

## Scope and workflow

- Write focused failure cases and unit tests before changing production code.
- Run `just check` and `just build` before finishing a task.
- Exercise the actual review, audit, or export commands when modifying data or evaluation code.
- Workstreams and branch coordination are recorded in [docs/data-expansion-workstreams.md](docs/data-expansion-workstreams.md) and `data/rewriter/workstreams.json`. Keep training rewards, rank ablations, and preference training deferred until data gates pass.

## Data freezing and holdout exclusion

- The source dataset lives in `data/rewriter/current`. Preserve all earlier training rows and the complete 100-row dev and 84-row validation source files.
- The active public export selects English rows, with frozen dev and validation splits bound by `data/rewriter/validation-freeze.json`, `data/rewriter/holdout-freeze.json`, and `data/rewriter/english-holdout-freeze.json`.
- Never edit frozen validation or dev splits, and never copy validation rows, fixtures, or source families into training expansions.
- Do not train on the retired 219-pair pilot (`data/rewriter/training/v3-20261027`).
- Keep benchmark rows limited to `input`, `output`, `mode`, and `checks`. Keep SFT rows limited to `system`, `user`, and `assistant`. Store source identity, context, literal rules, and review status in supporting files outside supervised text.
- Import only explicitly selected and redacted personal transcriptions into local ignored paths (`data/rewriter/personal/`). Never send raw personal exports to Git, Hugging Face, or hosted model APIs.

## Data review and evaluation rules

- Deterministic checks test literal and structural rules; they do not establish semantic quality. Authored good and bad fixture labels are test expectations, not independent human judgments.
- Comparisons use `openai/gpt-6-luna` as the cloud baseline and teacher, and `openai/gpt-6.1-sol` as the semantic judge. New synthetic data uses the pinned clean Arka pipeline with six Luna Decisions checks followed by exact-row GPT-6.1 Sol verification.
- Review agents may write verdicts and critiques only. They must not generate or edit examples or implement code.
- Every released row requires completed input, output, and pair verification with no unresolved mechanical failure.
- Bind all reviews to exact row hashes. Preserve rejected attempts, native vetoes, and uncertain or failed verdicts. Pending or failed judge calls never count as passes.
- Keep decision probabilities and LLM-judge verdicts separate from human gold labels. Mark all model-judge scores as diagnostic development-validation results.

## Hugging Face and Colab notebook rules

- Update the existing public dataset repositories (`jayshah5696/finetuning-labs-rewriter-training` and `jayshah5696/finetuning-labs-rewriter`) in place. Do not create duplicate dataset repositories.
- Public dataset repositories on Hugging Face contain only Parquet splits, dataset cards, and source-license notices. Keep code, notebooks, configs, review reports, and run logs in Git.
- Model repositories receive trained LoRA adapter weights, adapter config, tokenizer assets, and a model card.
- Keep one supported training notebook in Git: `notebooks/rewriter_train_l4.ipynb`. Do not copy notebooks into data releases.
- `scripts/build_lora_notebook.py` embeds the canonical `src/finetuning_labs/rewriter/lora_training.py` helpers into the notebook, and `scripts/check_lora_notebook.py` verifies AST and runtime parity.
- Public dataset and base-model downloads use `token=False`.
- The Colab notebook trains `LiquidAI/LFM2.5-230M` and `LiquidAI/LFM2.5-350M` with PEFT LoRA on Google Colab L4 GPUs. Each base model uses its own adapter repository (`jayshah5696/lfm2.5-230m-rewriter-lora` and `jayshah5696/lfm2.5-350m-rewriter-lora`).
- The notebook reads only two Colab Secrets: `HF_TOKEN` (for adapter upload) and `WANDB_API_KEY` (for loss logging). Do not launch Modal training jobs; Modal is used only for batch evaluation inference.
