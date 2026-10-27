# Human review priorities

The Sol verdicts are diagnostic. They have not been checked against independent human labels.
The current reference outputs and fixture labels remain drafts. No human labels were changed.

Review the six Luna failures against the complete input and candidate:

- `tw-repairs-04` retains the canceled request and "scratch that" instead of applying the repair.
- `tw-facts-05` ends the unfinished thought with a period. Confirm the intended punctuation rule.
- `tw-profiles-03` drops "We need three things" while keeping the three list items.
- `polish-05` also closes an unfinished thought with a period. Sol marks this critical but does
  not mark the similar transcribe failure critical. Align the critical-error classification.
- `long-prompt-02` adds "Do not perform the task" to a prompt whose purpose is to request a
  proposal. This is a new constraint that can prevent the downstream task from being done.
- `long-transcribe-02` preserves the opening "um okay" instead of removing the hesitation.

Review a blinded sample of passes too, including Liquid's two passes, `prompt-07` and
`polish-02`. Checking only failures cannot reveal false passes. Freeze the rubric after human
adjudication and reassess both arms if it changes.

The primary report counts every scheduled case once. It does not use setup calls as extra test
examples. The two setup calls are retained under `setup/`; their usage is separate from the
204 benchmark API calls. OpenRouter reported BYOK, so upstream cost and gateway charge differ.

All eight long cases come from four source documents. Prepare new independent families for the
locked test. Do not train on this calibration bundle or promote its current model judgments to
approved training targets. No fine-tuning has run.
