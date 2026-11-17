# Train and upload a rewriter LoRA

Open [rewriter_train_l4.ipynb](../notebooks/rewriter_train_l4.ipynb) in Google Colab
([open directly on Google Colab](https://colab.research.google.com/github/jayshah5696/finetuning-labs/blob/main/notebooks/rewriter_train_l4.ipynb)).
Choose a fresh L4 GPU runtime.
The first cell identifies the current notebook as `public-parquet-lora-current`.

Add exactly two secrets in Colab's Secrets panel and enable notebook access:

| Secret | Required access |
| --- | --- |
| HF_TOKEN | Write access to your Hugging Face model repository |
| WANDB_API_KEY | Your Weights & Biases API key |

The public dataset is readable without a paid Hugging Face plan. The HF token is needed
to upload your adapter. Keep both values in Secrets. Do not paste them into CONFIG.
Dataset, tokenizer and base-model downloads explicitly use `token=False`.

## Configure and run

Use the same notebook. CONFIG contains the model, dataset revision, adapter destination,
W&B project and LoRA settings. Set `MODEL_SIZE = "350m"` for the larger trial, or `"230m"`
to measure the data change on the previous base. The preset changes the model revision and
adapter destination together. Both presets use rank 16, alpha 32, dropout 0.05, three epochs,
learning rate 0.0002, batch size 1 and gradient accumulation 16. The context limit is 4,096.
Colab's CUDA PyTorch stays installed. The notebook installs the tested training libraries.

Change `hub_model_id` if another account owns your token. The default 350M destination is
`jayshah5696/lfm2.5-350m-rewriter-lora`. The 230M preset uses
`jayshah5696/lfm2.5-230m-rewriter-lora`. An adapter cannot be moved between those bases.
The `hub_private` setting controls new-repository visibility.

Run the cells in order:

1. Install libraries and read the two secrets.
2. Download train, dev and validation with `load_dataset`.
3. Train the LoRA adapter and log loss to W&B.
4. Evaluate validation loss, save the selected adapter and upload it.

Only adapter parameters receive updates. System and user tokens are masked in the training
loss. Assistant text and its EOS are supervised. Oversized examples raise an error instead of
being truncated. The current English export (`36a75c009d008415fad557d49005ba4e5c73f4d7`) has 1,979 train, 89 dev and 141 validation rows (expanded from the earlier 1,864 / 89 / 81 release at `00d59dda95867fe5bcb147b1b1c5a34b5ad7f071`). Checkpoint selection
uses dev loss. The final validation pass does not select a checkpoint.

Use a fresh runtime for this rerun. The notebook starts from the pinned base model and trains
a fresh adapter on all 1,979 English examples. It does not continue the previous adapter.
The dataset repository and split names stay the same. The last cell uploads the replacement
adapter to the configured model repository; earlier weights remain available in Hub commit history.
The notebook pins the updated data commit and checks its published split counts.

If you already have an earlier notebook copy open, keep that copy and update its existing CONFIG
after the original dictionary definition. The training helpers and train/upload cells are unchanged:

```python
CONFIG.update(
    model_id="LiquidAI/LFM2.5-350M",
    model_revision="9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
    dataset_revision="36a75c009d008415fad557d49005ba4e5c73f4d7",
    expected_split_counts={"train": 1979, "dev": 89, "validation": 141},
    hub_model_id="jayshah5696/lfm2.5-350m-rewriter-lora",
    run_name="lfm2.5-350m-rewriter-english",
    output_dir="/content/rewriter-lora-350m",
)
```

For 230M, use the 230M model, revision and destination from `configs/training-models.json`.
Keep the same updated dataset revision and split counts. Run from a fresh runtime after
changing these parameters.

## Output

The last cell uploads to a Hugging Face **model** repository. The final adapter folder contains:

- adapter_model.safetensors
- adapter_config.json
- Tokenizer and chat-template files
- README.md with the base model and training settings

The base model is still required when you load the adapter. Full base weights are not uploaded.
If upload fails, the adapter remains in `CONFIG["output_dir"]/final_adapter`. Correct the token
or repository name and rerun the upload cell.

W&B records training loss, dev loss and final `validation/loss`. Model artifacts go to Hugging Face;
W&B model uploads are disabled. These loss values are not a semantic rewrite pass rate. Use
the same Sol rubric on stock and adapted outputs for the later quality comparison.

## Verification boundary

The CPU integration executes the notebook's data, training, save and upload cells using public
Hub data. It substitutes a tiny random native LFM for the selected student, runs W&B offline and mocks
the final Hub model upload. It checks real adapter updates, unchanged base weights, adapter-only
serialization and exact adapter reload. It does not establish L4 memory use, BF16 execution,
online W&B authentication or the quality of a trained 230M or 350M model.

```sh
just notebook-check
```

The notebook is self-contained. It does not download code or review artifacts from Hugging Face.
It is the only supported notebook in Git. The builder embeds the Git helper module,
and the integration check verifies that the embedded code matches it. Older notebook copies
and packaging scripts are retained only in Git history.
