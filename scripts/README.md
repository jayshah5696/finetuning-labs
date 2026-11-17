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
- `verify_state_db_and_parquets.py` verifies 100% decision cache hits in `state.db` and exact
  byte parity across Arka Parquet stages for the expansion runs.
- `modal_eval.py::submit_lora` evaluates a pinned stock model and adapter on Modal. Select
  `--model-size 230m` or `--model-size 350m`. Use `--english-only` with the current public data.
  It performs inference only and saves each exact request and response to the evaluation volume.
- `judge_english_models.py` grades collected English runs with the existing Sol rubric and writes
  the comparison. It reuses a saved judgment only when its complete packet and judge settings
  match. It preserves failed and incomplete calls and does not count them as passes.
  Add `--hosted-run` for the saved Luna run. The combined CSV keeps every model's complete
  output and assessment beside the same input, reference, checks and context.

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
just notebook 36a75c009d008415fad557d49005ba4e5c73f4d7 350m
```

This rebuilds the same notebook. Both model presets select their base revision and adapter
destination together. Use `230m` for the data-only comparison on the previous base.

Verify the local English data without generating examples:

```sh
uv run --locked --group data --group hosted python scripts/check_english_release.py --tokenizers
just reviewed-audit
```

See [the run guide](../docs/colab-training.md) for the two Colab Secrets and upload configuration.

## Historical work

- `append_long_cases.py` (historical): authored the eight long calibration examples against the
  original 60-case bundle.
- `backfill_english_quality.py` (historical): recorded the three exclusions found by the targeted
  review and rebuilt the release from saved Arka artifacts and the Git baseline.
- `build_expansion_benchmark.py` (historical): built the 60-case `expansion-benchmark-v1` bundle
  (reads `all_20_cases.json` for the medium and long expansion cases).
- `check_training_cpu.py` (historical): checks historical full-training utilities locally. The
  former Colab runner is retired. Use the current LoRA notebook for training.
- `curate_arka_v2.py` (historical): applied recorded data corrections after Opus review while
  preserving earlier v2 artifacts.
- `merge_phase2_data.py` (historical): executed the one-off Phase 2 merge of `diverse-v2` and
  `validation-v1` into `current` and `benchmark-draft`.
- `recover_opus_report.py` (historical): recovered the omitted review array from a saved Opus
  reviewer tool command using AST literal parsing.
- `report_generation.py` (historical): generated mechanical-only reports for the initial stock
  Liquid baseline before semantic judging.
- `run_live_pipelines.py` (historical): ran the live Arka expansion pipelines and staged
  `diverse-v2` and `validation-v1`.
- `run_sol_reverification.py` (historical): re-ran live GPT-6.1 Sol verification on two repaired
  workspace-context scenarios (`sched-ws-med-02` and `sched-ws-med-06`).
- `sync_arka_decisions_and_stages.py` (historical): synced `DecisionClient` cache entries in
  `state.db` and Parquet stage files for `diverse-v2-run` and `validation-v1-run`.

The scripts that uploaded wheels, notebooks, review reports and raw evaluation runs to datasets
have been removed. The legacy Hub upload entry points refuse writes. Previous Git commits
retain their source and evidence.
The duplicate notebooks, private-data check script and old artifact-packaging scripts have
also been removed. Retrieve historical versions from Git when you need their original bytes.
