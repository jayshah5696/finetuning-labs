# Reproduce the English comparison

This run compares stock and uploaded LoRA models on the same 81 frozen English validation cases.
It does not train, upload, or change a model, the notebook, or the dataset. Generation runs on
Modal A10 GPUs with BF16 base weights, FP32 adapters, eager attention, seed 42, greedy decoding,
a 4,096-token context limit and a 2,048-token output budget. The generator receives the source
and mode instructions. It never receives the reference answer.

The 350M adapter at `b42778bcea7f4434365483e7b0ffd58e292fcead` lists the 1,864-row English
training data at `00d59dda95867fe5bcb147b1b1c5a34b5ad7f071`. The available 230M adapter at
`ba71856780e4d2398218fe76756b3390556437ea` is unchanged from the earlier 1,500-row run at
`139bb98b0500fa12ad43dc6c8c5cd4d17645eb86`. Do not describe the 230M adapter as retrained on
the latest data. The current evaluation uses the same English validation cases for both.

```sh
uv run --locked --group data --group remote modal run --detach scripts/modal_eval.py::submit_lora \
  --run-id english-230m-validation-20261029-a10g --gpu A10G --model-size 230m --english-only \
  --adapter-revision ba71856780e4d2398218fe76756b3390556437ea \
  --dataset-revision 00d59dda95867fe5bcb147b1b1c5a34b5ad7f071
uv run --locked --group data --group remote modal run --detach scripts/modal_eval.py::submit_lora \
  --run-id english-350m-validation-20261029-a10g --gpu A10G --model-size 350m --english-only \
  --adapter-revision b42778bcea7f4434365483e7b0ffd58e292fcead \
  --dataset-revision 00d59dda95867fe5bcb147b1b1c5a34b5ad7f071
uv run --locked --group data --group remote modal run scripts/modal_eval.py::collect \
  --run-id english-230m-validation-20261029-a10g
uv run --locked --group data --group remote modal run scripts/modal_eval.py::collect \
  --run-id english-350m-validation-20261029-a10g
```

The two `modal.json` receipts contain app IDs, call IDs and exact generation configurations.
The volume is `finetuning-labs-evals`. Each run ID is its volume root. Collection waits for the
recorded function call, verifies that it returned, and downloads its durable artifacts.
Use new run IDs to repeat inference. Existing run IDs cannot be silently overwritten.

With the authorized OpenRouter credential in the process environment, grade the saved outputs:

```sh
uv run --locked --group hosted python scripts/judge_english_models.py \
  --run data/rewriter/evals/english-230m-validation-20261029-a10g \
  --run data/rewriter/evals/english-350m-validation-20261029-a10g \
  --output data/rewriter/evals/english-comparison-20261029 --workers 4
```

Sol uses the unchanged five-criterion rubric through Arka and OpenRouter. Generator identity
is hidden from its packet. A pass requires all semantic criteria and declared deterministic
checks to pass. Missing, failed, uncertain, or truncated cases stay in the denominator and
never become passes. A critical error changes, invents, or omits intended meaning. Counts
describe outputs with at least one critical error, rather than the number of individual errors.
Truncated outputs are not semantically judged and are reported separately.

Exact cached judgments retain their original response bytes and timestamps. Each reuse receipt
records its source and request/result hashes. Completed judgments are not rerolled. API usage
in cached responses describes the original call, not incremental spend during this comparison.
When several historical judgments match one packet, the index uses the first source in lexical
path order. This differs from the earlier 1,500-row run for `tw-facts-05`. Its period after an
unfinished sentence received both a pass and a mode failure in earlier Sol calls. The current
comparison retains the failure. The unchanged 230M adapter therefore scores 50/81 here rather
than the earlier 51/81. See `historical-judge-disagreement.json`. This is judge disagreement,
not evidence that the model changed. No verdict was selected by favorability or requested again.

The `runtime/` text files preserve the exact generator and submitter sources used for the GPU
run. Later source formatting changes do not alter the recorded generation runner hash.
The 81 cases are development validation used in earlier iterations. Model judgments and this
small reused validation set do not establish human-verified accuracy on unseen inputs.
