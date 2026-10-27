# Finetuning Labs

Use `uv` and the project Makefile. Keep this project independent of the root uv workspace.

The current release is reviewed v2 with 65 training, 17 development and 84 validation pairs.
Keep all v2 validation rows and source families out of future training expansions. The student
is `LiquidAI/LFM2.5-230M`. Comparisons use GPT-6 Luna as the cloud baseline and teacher,
and GPT-6.1 Sol as the historical baseline evaluator. Replacement data uses the latest clean Arka
pipeline with Luna Decisions, followed by Opus verification. GPU work targets Google Colab L4.

Opus is restricted to data verification. It may write row verdicts and critiques. It must not
generate or edit examples, implement code, or review implementation. The parent and Luna handle
corrections. Every released exact row needs input, output, and pair verification, with no unresolved
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
The user authorizes Git commits and pushes, private shared-data Hugging Face uploads, diagnostic
evaluations, and preparation of separate training data. GPU training remains a later step.
Human calibration is still pending; mark model scores diagnostic.
