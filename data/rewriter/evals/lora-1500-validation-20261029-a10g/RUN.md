# Reproduce the 1,500-example adapter evaluation

The same frozen 84 development-validation cases and unchanged Sol rubric are used in both arms.
Generation runs on one Modal A10 with BF16 base weights and FP32 LoRA tensors, as in the previous
comparison. This run does not train a model or modify the dataset or notebook.

```sh
uv run --locked --group data --group remote modal run --detach scripts/modal_eval.py::submit_lora \
  --run-id lora-1500-validation-20261029-a10g --gpu A10G \
  --adapter-revision ba71856780e4d2398218fe76756b3390556437ea \
  --dataset-revision 139bb98b0500fa12ad43dc6c8c5cd4d17645eb86
uv run --locked --group data --group remote modal run scripts/modal_eval.py::collect \
  --run-id lora-1500-validation-20261029-a10g
```

The Modal app is ap-Z7blHmgYnKoCzid1ThVYUx, function call fc-01M634CQDW6Y7Z2AVRWYVBSWTP,
volume finetuning-labs-evals and volume root lora-1500-validation-20261029-a10g. The local receipt
is modal.json. The pinned configuration and runner hash are recorded there. The stock arm has
no adapter. Both arms use the exact same prompts and greedy generation configuration.

The authorized OpenRouter key is loaded into process memory from the existing shell config.
No credential is written to these records. Use OPENROUTER_API_KEY in your own environment.

```sh
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  --bundle data/rewriter/reviewed-v2/validation-bundle judge \
  --generation data/rewriter/evals/lora-1500-validation-20261029-a10g/stock \
  --output data/rewriter/evals/lora-1500-validation-20261029-a10g/stock-judge --workers 4
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  --bundle data/rewriter/reviewed-v2/validation-bundle judge \
  --generation data/rewriter/evals/lora-1500-validation-20261029-a10g/lora \
  --output data/rewriter/evals/lora-1500-validation-20261029-a10g/lora-judge --workers 4
uv run --locked python -m finetuning_labs.rewriter.comparison \
  --bundle data/rewriter/reviewed-v2/validation-bundle \
  --run data/rewriter/evals/lora-1500-validation-20261029-a10g --left stock --right lora
```

No model output receives the reference answer during generation. Saved generation and judge
calls bind the exact prompts, outputs, configurations and hashes. Only Sol evaluates outputs.
Existing saved calls are verified, not implicitly reissued. Human calibration and an independent
unseen final test remain pending. Compare quality counts, not training loss, with the previous
1,000-example adapter result.

## Exact judgment reuse

The stock outputs are byte-identical to the previous run. Its 84 Sol calls were reused only after
matching the complete expected packet, judge model, settings and rubric, and validating their
request/result hashes. Another 41 updated-adapter outputs match an existing exact packet and
reuse that judgment. The other 42 complete updated-adapter outputs receive fresh Sol calls.
One updated-adapter output is truncated and remains a failure without a semantic judgment.
`judge-reuse.json` records every source path and request/result hash. Cached raw records retain
their original bytes and timestamps. This avoids changing judgments for unchanged outputs.
The comparison's API usage totals include cached historical records, not only incremental spend.
