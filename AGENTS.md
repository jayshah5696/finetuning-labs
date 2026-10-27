# Finetuning Labs

Use `uv` and the project Makefile. Keep this project independent of the root uv workspace.

The current scope includes the accepted development-validation set and a separate training-data
pilot. The student is `LiquidAI/LFM2.5-230M`. Comparisons use GPT-6 Luna as the cloud baseline and teacher,
and GPT-6.1 Sol as the semantic evaluator. GPU work targets Google Colab L4.

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
The user authorizes Git commits and pushes, private shared-data Hugging Face uploads, diagnostic
evaluations, and preparation of separate training data. GPU training remains a later step.
Human calibration is still pending; mark model scores diagnostic.
