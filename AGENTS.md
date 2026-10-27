# Finetuning Labs

Use `uv` and the project Makefile. Keep this project independent of the root uv workspace.

The current scope is a reviewable rewriter calibration draft and its offline checks. The student
is `LiquidAI/LFM2.5-230M`. Later comparisons use GPT-6 Luna as the cloud baseline and teacher,
and GPT-6.1 Sol as the semantic evaluator. GPU work targets Google Colab L4.

Keep benchmark rows limited to `input`, `output`, `mode`, and `checks`. Store source identity,
context, literal rules, and review status in supporting files. Never put those records into
supervised text. Preserve the original guide's `system`, `user`, and `assistant` export.

Write failure cases and focused tests before production code. Run `make check` and `make build`.
Exercise the actual review and export commands before reporting completion.

Deterministic checks do not establish semantic quality. Good and bad fixture labels are authored
expectations, not independent human judgments. Pending or failed judge calls never count as passes.

Keep this bundle draft until the user reviews the cases. Do not train on it or use its source
families in a locked test. Import only selected personal transcriptions. Keep private exports out
of Git. The user now authorizes committing this project, a private shared-data Hugging Face
upload, and diagnostic Liquid/Luna evaluations with a shared Sol judge. Training and Git pushes
remain outside this step. Human calibration is still pending; mark model scores diagnostic.
