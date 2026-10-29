# Targeted data expansion

The original adapter comparison identified failure categories. New fictional episodes target
those categories, not the validation texts. The existing public training dataset is updated
in place. Dev and validation remain frozen.

Arka is pinned to `5d353740ae1dd34742841071752814028db1489a`. Generation uses
`openai/gpt-6-luna`, six native criteria use `openai/gpt-6-luna-decisions`, and every new
data review uses `openai/gpt-6.1-sol`. Existing exact Opus reviews remain attached to the
original rows. Review verdicts do not represent human labels.

Run with OPENROUTER_API_KEY in the environment. Never paste credentials into a command or file.
The authoring command is stochastic; the committed evidence preserves the actual selected
examples and rejected attempts. Reusing a run directory preserves its saved plans and calls.

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.boost_data \
  --run artifacts/fidelity-boost --batch-size 60
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.boost_data \
  --run artifacts/fidelity-boost --finish
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.boost_data \
  --run artifacts/fidelity-boost --apply
make reviewed-audit
make check
make build
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.hf_data --publish
```

Supplemental long-prompt and long-transcription batches were prepared while the initial run completed. Their
separate source plans and all attempts are preserved with the other batch evidence. The second
supplement uses a request plus a verbatim reference-notes block. The same native and Sol gates
apply. Supplementary source IDs cannot collide with the initial source plan.

Rebuild the notebook with the training revision from `releases/public-parquet.json`:

```sh
uv run --locked python scripts/build_lora_notebook.py --dataset-revision TRAINING_COMMIT
uv run --no-project --python /tmp/rewriter-training-check/bin/python python \
  scripts/check_lora_notebook.py --output artifacts/fidelity-notebook-check
uv run --locked --group data --group hosted python scripts/check_public_hub.py \
  --output artifacts/fidelity-public-viewer-check.json
```

The CPU check uses all public data for encoding, a tiny random LFM for four optimizer steps,
offline W&B, and a mocked adapter upload. It verifies the notebook workflow. It does not train
the actual 230M model or establish semantic improvement. The user runs the final L4 notebook,
then the new adapter is compared on the same frozen validation set and Sol rubric.
