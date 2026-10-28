# Finetuning Labs

Fine-tune a small rewriter with a LoRA adapter. The default model is LiquidAI/LFM2.5-230M.
Code and notebooks stay in Git. Hugging Face hosts the public data and trained adapters.

## Latest trained adapter

The [230M LoRA](https://huggingface.co/jayshah5696/lfm2.5-230m-rewriter-lora) is trained and uploaded.
On the frozen 84-case diagnostic, stock passes 4/84 and the LoRA passes 48/84. The LoRA preserves
all eight explicit identity cases, but has 25 judge-flagged critical errors and passes only
3/13 long cases. It is not ready for acceptance under the current targets.
[Read the findings](data/rewriter/evals/lora-validation-20261028-a10g/FINDINGS.md) and
[the full report](data/rewriter/evals/lora-validation-20261028-a10g/comparison/REPORT.md).
Both arms used the same A10 runtime and Sol rubric. These are development-validation model judgments.

## Data

[Browse the training dataset](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/viewer/default/train).

| Split | Rows | Use |
| --- | ---: | --- |
| train | 1,000 | Fit the LoRA adapter |
| dev | 100 | Select the checkpoint |
| validation | 84 | Evaluate after training |

Each mode has 250 training and 25 dev pairs. The 84 validation examples are unchanged.
The original [68-case calibration benchmark](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter/viewer/default/validation)
is also public and remains separate.

The default Viewer configuration has input, output, mode and checks. The `sft` configuration
has system, user and assistant, including the required language, profile and dictionary context.
Both use Parquet. The Hub repositories contain data, dataset cards and source-license notices.
Model reviews are not human labels. No personal transcription history is included.

```python
from datasets import load_dataset

data = load_dataset("jayshah5696/finetuning-labs-rewriter-training", "sft")
```

## Train in Colab

Open [the notebook in Colab](https://colab.research.google.com/github/jayshah5696/jadoo-labs/blob/main/projects/finetuning-labs/notebooks/rewriter_train_l4.ipynb)
in a fresh L4 runtime. The [notebook file](notebooks/rewriter_train_l4.ipynb) is in Git.
Its first cell shows version `public-parquet-lora-2`. This is the only supported notebook in
the project. Public dataset and base-model downloads use `token=False`.
Add two Colab Secrets and enable access:

- `HF_TOKEN`, with write access to the destination model repository.
- `WANDB_API_KEY`, for Weights & Biases logging.

Edit CONFIG if you want a different model, adapter repository or training settings. Then run
the notebook cells in order. It downloads the data, trains LoRA, logs loss to W&B, saves the
selected adapter and uploads it to a Hugging Face model repository. The default destination is
`jayshah5696/lfm2.5-230m-rewriter-lora`. Change the owner if you use another account.

The final folder contains adapter weights, adapter config, tokenizer files and a model card.
It does not merge or upload the base model. No notebook or source code goes to the Hub.
See [COLAB_TRAINING.md](COLAB_TRAINING.md) for the short run guide.

## Check and publish

```sh
uv sync --locked
make check
make build
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.hf_data
```

The last command prepares the public Parquet files locally. Add `--publish` to update the
two project dataset repositories. It removes non-data files from their current trees.
The older artifact-upload commands are retired.

The notebook helpers are in `src/finetuning_labs/rewriter/lora_training.py`.
The notebook builder and CPU integration check are in `scripts/`.
[TRAINING_DATA.md](TRAINING_DATA.md) records data sources and review history.
[VALIDATION.md](VALIDATION.md) defines later semantic scoring. Training and validation loss
are not rewrite accuracy. The uploaded adapter has now completed the paired GPU diagnostic above.
