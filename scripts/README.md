# Finetuning Labs scripts

Code and verification records stay in Git. Hugging Face dataset repositories contain Parquet
data and source notices. Model repositories receive the trained adapter and tokenizer artifacts.

## Current workflow

- `build_lora_notebook.py` builds the self-contained Colab notebook from the tested helpers.
- `check_lora_notebook.py` runs the real notebook cells with public data and a tiny random CPU LFM.
  W&B runs offline and the final model upload is mocked. The base weights stay frozen.
- `check_public_hub.py` verifies public file hashes, all downloaded rows and every Viewer split
  without authentication. It compares them with the unchanged Git source rows.
- `data_distribution.py` measures wording repetition and coverage. It does not assign meaning quality.
- `check_english_release.py` checks the exact 500 additions, anchors, code and exclusion mapping.
  With `--tokenizers`, it encodes every English row with both pinned model tokenizers.
- `backfill_english_quality.py` records the three exclusions found by the final targeted review.
  It uses saved Arka artifacts and a pristine Git baseline, audits a rebuilt release and swaps it
  into place. It makes no new model calls and is not part of Colab training.

Publish the public data with the dataset-only module:

```sh
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.hf_data
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.hf_data --publish
```

The first command prepares files under ignored artifacts. The second updates the two project
dataset repositories, removes non-data files from their current trees and enables public viewing.
The source reviewed rows stay unchanged. Publication receipts stay in Git.

Rebuild the notebook using the current public-data revision:

```sh
uv run --locked python scripts/build_lora_notebook.py \
  --dataset-revision 00d59dda95867fe5bcb147b1b1c5a34b5ad7f071 --model-size 350m
```

This rebuilds the same notebook. Both model presets select their base revision and adapter
destination together. Use `230m` for the data-only comparison on the previous base.

Verify the local English data without generating examples:

```sh
uv run --locked --group data --group hosted python scripts/check_english_release.py --tokenizers
make reviewed-audit
```

See [the run guide](../COLAB_TRAINING.md) for the two Colab Secrets and upload configuration.

## Historical work

`check_training_cpu.py` checks historical full-training utilities locally. The former Colab
runner is retired. Use the current LoRA notebook for training.

The scripts that uploaded wheels, notebooks, review reports and raw evaluation runs to datasets
have been removed. The legacy Hub upload entry points refuse writes. Previous Git commits
retain their source and evidence.
The duplicate notebooks, private-data check script and old artifact-packaging scripts have
also been removed. Retrieve historical versions from Git when you need their original bytes.
