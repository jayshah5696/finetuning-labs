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
  --dataset-revision bbcdb7692445a4310296060606b8a6dcce439f47
```

See [the run guide](../COLAB_TRAINING.md) for the two Colab Secrets and upload configuration.

## Historical work

`check_training_cpu.py` and `release_scaled_data.py` record the earlier full-training workflow
and reviewed-data selection. They are retained for local reproduction. Their old artifact
packages are not the current Hub layout.

The scripts that uploaded wheels, notebooks, review reports and raw evaluation runs to datasets
have been removed. The legacy Hub upload entry points refuse writes. Previous Git commits
retain their source and evidence.
