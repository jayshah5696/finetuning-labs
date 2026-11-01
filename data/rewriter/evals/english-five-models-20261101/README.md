# Five-model English rewrite comparison

This run evaluates both Liquid bases, their newly uploaded LoRA adapters, and GPT-6 Luna.
The benchmark is development validation, not an unseen final test. Model reviews are not
independent human labels. The exact source bundle and runner snapshots are retained here.

## Inputs and models

- Dataset: `jayshah5696/finetuning-labs-rewriter-training`, commit
  `36a75c009d008415fad557d49005ba4e5c73f4d7`.
- Evaluation: 141 English validation cases. The original 81 cases retain their exact text,
  context, rules and order. The other 60 cases are the newly appended expansion.
- 230M base: `LiquidAI/LFM2.5-230M`, commit
  `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
- 230M adapter: `jayshah5696/lfm2.5-230m-rewriter-lora`, commit
  `664f541045c0148e7bec66b267b35238716d4688`.
- 350M base: `LiquidAI/LFM2.5-350M`, commit
  `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
- 350M adapter: `jayshah5696/lfm2.5-350m-rewriter-lora`, commit
  `b660a2046092b07e46061ee25b738da3f4ef0688`.
- Cloud generation: `openai/gpt-6-luna`, reasoning disabled and a 2,048-token output limit.
- Judge: `openai/gpt-6.1-sol`, high reasoning, the existing unchanged rubric and a
  16,384-token response limit. All five arms use the same judge settings.

Both adapter cards identify the new 1,979-row English training split, the 89-row dev split,
rank 16 and alpha 32. Their cards, configs and safetensors hashes are in `model-cards/`.
Those metadata records identify the runs; they are not a reproduction of training.

Liquid inference runs on Modal NVIDIA A10 GPUs. Base and adapter settings match within each
size. Both use BF16 base weights, greedy generation, eager attention, seed 42, a 4,096-token
context limit and a 2,048-token output limit. PEFT adapter dtypes and the actual generation
configuration are recorded with every output. No training is part of this evaluation.

## Scoring

A strict pass needs the deterministic case rules and every Sol criterion to pass. The
criteria cover meaning, terms and literals, mode, readability and output discipline.
The denominator includes every scheduled case. Uncertain, failed, interrupted, truncated
and erroneous calls never count as passes.

A critical error means Sol identified at least one changed, omitted or invented fact or
meaning in that case. The count is cases with errors, not individual defects. Review
coverage and transport errors must accompany that count.

The report also retains Sol-only pass counts and cases where Sol passes but a deterministic
rule fails. Some such rules intentionally require exact identity, file tags or list format.
Do not silently waive them or interpret every rule-only failure as a meaning error.

Every model receives the same case-level system and user text. Generation does not receive
the reference. The judge receives the input, candidate, reference, mode, context and checks,
without generator identity. Reports verify the exact saved request/result hashes, rather
than trusting edited metrics files. Existing failed calls are not implicitly retried.

`comparison.json` reports all 141 cases, the original 81 and the new 60 separately. It also
includes per-mode, long-case, identity and paired base-versus-adapter results.
`case-results.csv` puts all five outputs and their verdicts alongside each input/reference.
The manifest's long-case bucket is reported separately; its other bucket includes medium
cases and must not be described as exclusively short.

## Run records and reproduction

GPU receipts, generations and saved calls are in the sibling directories:

- `../english-230m-validation-20261101-a10g/`
- `../english-350m-validation-20261101-a10g/`

The Modal app name is `finetuning-labs`. Durable outputs use volume
`finetuning-labs-evals` under those run names. Each `modal.json` records the app and
function-call IDs. Collection verifies completion and copies outputs locally.

The launch parameters were:

```sh
uv run --locked --group remote --group data --group hosted modal run --detach scripts/modal_eval.py::submit_lora --run-id english-230m-validation-20261101-a10g --gpu A10G --adapter-revision 664f541045c0148e7bec66b267b35238716d4688 --dataset-revision 36a75c009d008415fad557d49005ba4e5c73f4d7 --model-size 230m --english-only --validation-source current
uv run --locked --group remote --group data --group hosted modal run --detach scripts/modal_eval.py::submit_lora --run-id english-350m-validation-20261101-a10g --gpu A10G --adapter-revision b660a2046092b07e46061ee25b738da3f4ef0688 --dataset-revision 36a75c009d008415fad557d49005ba4e5c73f4d7 --model-size 350m --english-only --validation-source current
```

Do not resubmit those run IDs. Use the `collect` entry point with the saved run ID to retrieve
the existing calls. A new generation experiment needs a fresh run ID.

`runtime/run_hosted.py.txt` contains the exact hosted launcher. It uses the existing Arka
client and evaluator. The authorized OpenRouter key is read into memory, never saved with
the run. After all outputs and judgments exist, recompute the comparison offline:

```sh
uv run --locked --group hosted --group data python data/rewriter/evals/english-five-models-20261101/report.py
```

The runner's source commit is recorded as dirty because unrelated data-publication work is
present in the shared checkout. Exact runtime snapshots and hashes record what this run used.
