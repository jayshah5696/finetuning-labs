# Train and upload a rewriter LoRA

Open [rewriter_train_l4.ipynb](notebooks/rewriter_train_l4.ipynb) in Google Colab.
Choose a fresh L4 GPU runtime.

Add exactly two secrets in Colab's Secrets panel and enable notebook access:

| Secret | Required access |
| --- | --- |
| HF_TOKEN | Write access to your Hugging Face model repository |
| WANDB_API_KEY | Your Weights & Biases API key |

The public dataset is readable without a paid Hugging Face plan. The HF token is needed
to upload your adapter. Keep both values in Secrets. Do not paste them into CONFIG.

## Configure and run

CONFIG contains the model, dataset revision, adapter destination, W&B project and LoRA settings.
When you change the base model, update both `model_id` and `model_revision`.
The defaults are LiquidAI/LFM2.5-230M, rank 16, alpha 32, dropout 0.05, three epochs,
learning rate 0.0002, batch size 1 and gradient accumulation 16. The context limit is 4,096.
Colab's CUDA PyTorch stays installed. The notebook installs the tested training libraries.

Change `hub_model_id` if another account owns your token. The default destination is
`jayshah5696/lfm2.5-230m-rewriter-lora`. The `hub_private` setting controls new-repository visibility.

Run the cells in order:

1. Install libraries and read the two secrets.
2. Download train, dev and validation with `load_dataset`.
3. Train the LoRA adapter and log loss to W&B.
4. Evaluate validation loss, save the selected adapter and upload it.

Only adapter parameters receive updates. System and user tokens are masked in the training
loss. Assistant text and its EOS are supervised. Oversized examples raise an error instead of
being truncated. Train has 1,000 rows, dev has 100 and validation has 84. Checkpoint selection
uses dev loss. The final validation pass does not select a checkpoint.

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
Hub data. It substitutes a tiny random native LFM for the 230M model, runs W&B offline and mocks
the final Hub model upload. It checks real adapter updates, unchanged base weights, adapter-only
serialization and exact adapter reload. It does not establish L4 memory use, BF16 execution,
online W&B authentication or the quality of a trained 230M model.

```sh
uv run --no-project --python /tmp/rewriter-training-check/bin/python python \
  scripts/check_lora_notebook.py --output artifacts/lora-check-new
```

The notebook is self-contained. It does not download code or review artifacts from Hugging Face.
