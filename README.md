# Finetuning Labs

Fine-tune an English rewriter with a LoRA adapter. The same notebook supports LiquidAI/LFM2.5-230M
and LiquidAI/LFM2.5-350M. The larger trial defaults to 350M.
Code and notebooks stay in Git. Hugging Face hosts the public data and trained adapters.

This project now lives in [jayshah5696/finetuning-labs](https://github.com/jayshah5696/finetuning-labs).
The repository is private. [Migration notes](docs/repository-migration.md) describe the preserved
history and the original checkout.

[Read the English experiment log](https://sangam.jay-hirajoshi.ts.net/p/finetuning-labs-experiment-log)
for the setup, data iterations, Arka review process and matched Luna comparison.

## Latest English comparison

All five models now use the same 81 English validation cases, prompts, output budget and Sol rubric.
Stock 230M passes 4/81, its uploaded LoRA 50/81, stock 350M 18/81, the new 350M LoRA 62/81,
and GPT-6 Luna 73/81. Luna has one judge-flagged critical-error output and no truncations.
[Read the full five-model report](data/rewriter/evals/english-all-models-20261029/REPORT.md) or
[compare every output beside its source](data/rewriter/evals/english-all-models-20261029/case-reviews.csv).
The earlier 64-case Luna comparison remains historical evidence. The current comparison includes
all 81 cases. Liquid uses greedy decoding; hosted Luna uses provider-default sampling with reasoning off.

The uploaded [350M LoRA](https://huggingface.co/jayshah5696/lfm2.5-350m-rewriter-lora) passes
62/81 frozen English validation cases, compared with 18/81 for stock 350M. It has 13
judge-flagged critical-error outputs and no truncations. The available 230M LoRA passes
50/81, compared with 4/81 for stock 230M. Its upload still records the older 1,500-row data;
the new 350M upload records 1,864 English rows. This comparison changes data and model size.
Both models preserve 7/8 identities. Long passes are 4/13 for the 230M LoRA and 7/13 for 350M.
[Read the four-arm report](data/rewriter/evals/english-comparison-20261029/REPORT.md).
These are Sol judgments on development validation. Critical errors remain in canceled requests,
spoken corrections, prohibitions and unfinished thoughts. The earlier 230M score of 51/81
differs by one conflicting historical judgment, detailed in the new report. Its weights and
outputs are unchanged.

## Earlier 230M comparison

The [230M LoRA](https://huggingface.co/jayshah5696/lfm2.5-230m-rewriter-lora) was retrained on all
1,500 examples. The checked revision is `ba71856780e4d2398218fe76756b3390556437ea`.
On the frozen 84-case diagnostic, stock passes 4/84 and the updated LoRA passes 53/84, compared
with 48/84 for the previous adapter. Long passes improve from 3/13 to 4/13, but identity falls
from 8/8 to 7/8. It has 18 judge-flagged critical-error outputs among 83 complete outputs and
one truncation. It is not ready for acceptance under the current targets.
[Read the findings](data/rewriter/evals/lora-1500-validation-20261029-a10g/FINDINGS.md) and
[the full report](data/rewriter/evals/lora-1500-validation-20261029-a10g/comparison/REPORT.md).
Both arms ran on the same A10 with the unchanged Sol rubric. Exact judgments are reused only
for unchanged packets. These are development-validation model judgments, not human gold labels.

The English subset is the comparison for the next run. The current adapter passes 51/81,
compared with 46/81 for the previous adapter. Seventeen of 80 complete English outputs have
judge-flagged critical errors; one more output truncated. Errors include reversed recipients,
amounts and prohibitions, lost correction context, invented code and missing long sections.
[Read the English cases](data/rewriter/evals/lora-1500-validation-20261029-a10g/ENGLISH.md).

## Data

[Browse the training dataset](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter-training/viewer/default/train).

| Split | Rows | Use |
| --- | ---: | --- |
| train | 1,864 | Fit the LoRA adapter |
| dev | 89 | Select the checkpoint |
| validation | 81 | Evaluate after training |

The active public export is English only. It selects 1,364 earlier English training pairs and
adds 500 new Arka episodes targeting the remaining critical errors. The additions include
100 long examples and 125 unchanged-input examples. Recipient and numeric corrections use
both directions so the correct answer cannot be guessed from a fixed ordering.
The full 2,000-row source training pool and the original 100/84 source holdouts remain in Git.
Earlier training rows are unchanged. The English dev and validation texts and order are frozen.
The same public dataset is updated in place.
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

Open [the notebook in Colab](https://colab.research.google.com/github/jayshah5696/finetuning-labs/blob/main/notebooks/rewriter_train_l4.ipynb)
in a fresh L4 runtime. The [notebook file](notebooks/rewriter_train_l4.ipynb) is in Git.
While this repository is private, Colab needs GitHub access. You can instead download the notebook
and use Colab's File > Upload notebook command.
Its first cell shows version `public-parquet-lora-current`. This is the only supported notebook in
the project. Public dataset and base-model downloads use `token=False`.
Add two Colab Secrets and enable access:

- `HF_TOKEN`, with write access to the destination model repository.
- `WANDB_API_KEY`, for Weights & Biases logging.

Set `MODEL_SIZE` to `350m` or `230m` in CONFIG. The preset selects the base revision and its
matching adapter repository. Then run
the notebook cells in order. It downloads the data, trains LoRA, logs loss to W&B, saves the
selected adapter and uploads it to a Hugging Face model repository. The default destination is
`jayshah5696/lfm2.5-350m-rewriter-lora` for 350M and
`jayshah5696/lfm2.5-230m-rewriter-lora` for 230M. Change the owner if you use another account.
You run training in Colab; this data update does not train or upload a new adapter.

The final folder contains adapter weights, adapter config, tokenizer files and a model card.
It does not merge or upload the base model. No notebook or source code goes to the Hub.
See [COLAB_TRAINING.md](COLAB_TRAINING.md) for the short run guide.

## Check and publish

```sh
git clone https://github.com/jayshah5696/finetuning-labs.git
cd finetuning-labs
uv sync --locked --group data --group hosted
make check
make build
uv run --locked --group data --group hosted python -m finetuning_labs.rewriter.hf_data
```

Checks use the data and hosted dependency groups. Export `OPENROUTER_API_KEY` before commands
that call hosted models. Keep credentials in an untracked local `.env` file if needed, and load
them into the shell before running those commands. No credentials moved from the old repository.

The last command prepares the public Parquet files locally. Add `--publish` to update the
two project dataset repositories. It removes non-data files from their current trees.
The older artifact-upload commands are retired.

The notebook helpers are in `src/finetuning_labs/rewriter/lora_training.py`.
The notebook builder and CPU integration check are in `scripts/`.
[TRAINING_DATA.md](TRAINING_DATA.md) records data sources and review history.
[VALIDATION.md](VALIDATION.md) defines later semantic scoring. Training and validation loss
are not rewrite accuracy. The uploaded adapter has now completed the paired GPU diagnostic above.
