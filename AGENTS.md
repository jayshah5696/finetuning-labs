# Finetuning Labs

Use `uv` and the project Makefile. Keep this project independent of the root uv workspace.

The current data-first work is split across the threads in
`docs/data-expansion-workstreams.md` and `data/rewriter/workstreams.json`. Active workers own
separate worktrees and staging namespaces. The coordinating thread owns release integration.
Keep training rewards, rank ablations and preference training deferred until data readiness.

The source dataset is data/rewriter/current. Preserve all earlier training rows and the complete
100-row dev and 84-row validation files. The active public export selects English only, with
frozen 89-row dev and 81-row validation views. English additions append only to training.
Update the existing public dataset in place. Do not introduce a new dataset version or repository.
The public Hub repositories now contain Parquet data, dataset cards and source-license notices
only. Code, notebooks, configs, review reports and run logs belong in Git. Do not upload those
artifacts to Hugging Face dataset repositories. Model repositories receive trained adapter weights,
adapter config, tokenizer assets and a model card. The active Colab notebook uses PEFT LoRA and
only two Colab Secrets: HF_TOKEN with model-write access, and WANDB_API_KEY. It loads public data
with load_dataset, logs loss to W&B, and uploads the final adapter. Do not restore the retired
private artifact-bundle workflow. The user explicitly authorized public synthetic-data visibility.
Keep exactly one supported notebook in Git: notebooks/rewriter_train_l4.ipynb. Do not copy notebooks
into data releases. The builder embeds the canonical lora_training.py helpers, and the checker
compares their ASTs. Public data and base-model downloads use token=False. Old notebooks and
runtime bundles are available in Git history. Data review must not depend on those code bundles.
Keep all v2 validation rows and source families out of future training expansions. The student
models are `LiquidAI/LFM2.5-230M` and `LiquidAI/LFM2.5-350M`. Use the same notebook with its
model preset; each base needs its own adapter repository. The user will run training in Colab.
Do not launch Modal training. Comparisons use GPT-6 Luna as the cloud baseline and teacher,
and GPT-6.1 Sol as the historical baseline evaluator. Replacement data uses the latest clean Arka
pipeline with Luna Decisions, followed by GPT-6.1 Sol verification. GPU work targets Google Colab L4.

Existing data retains its exact Opus verification. After an OpenRouter credit-limit failure,
the user changed the remaining review and fix loop to GPT-6.1 Sol only. Use Sol for new reviews.
Review agents may write verdicts and critiques. They must not generate or edit examples or
implement code. The parent handles corrections. Every released exact row needs input, output,
and pair verification, with no unresolved
mechanical failure. Keep decision scores separate from Opus verdicts and from human labels.
Preserve rejected attempts and bind all reviews to exact row hashes. Do not train on the retired
219-pair pilot. Publish corrected validation references as a new version of the original 68 cases.

Keep benchmark rows limited to `input`, `output`, `mode`, and `checks`. Store source identity,
context, literal rules, and review status in supporting files. Never put those records into
supervised text. Preserve the original guide's `system`, `user`, and `assistant` export.

Write failure cases and focused tests before production code. Run `make check` and `make build`.
Exercise the actual review and export commands before reporting completion.

Deterministic checks do not establish semantic quality. Good and bad fixture labels are authored
expectations, not independent human judgments. Pending or failed judge calls never count as passes.

The user accepted the 68 cases for development validation on 27 October 2026. Preserve their
bytes and exclude their source families and fixtures from training. Acceptance of this dataset
does not supply independent human labels for model outputs or establish a locked final test.
Import only selected personal transcriptions. Keep personal exports out of Git and hosted calls.
The user authorizes Git commits and pushes, public shared-data Hugging Face uploads, diagnostic
evaluations, and preparation of separate training data. The user has trained and uploaded the
230M LoRA. Use its pinned revision for diagnostic comparison; do not retrain it as part of evaluation.
Human calibration is still pending; mark model scores diagnostic.
