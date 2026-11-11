# Repository migration

The standalone repository is https://github.com/jayshah5696/finetuning-labs.
It starts private. Changing GitHub visibility later is a separate action.

## Source and history

The project came from `projects/finetuning-labs` in
https://github.com/jayshah5696/jadoo-labs at commit
`0b0bea6652f0aaaec5add7c9ce23807d7e0300b0`.

The extraction preserves all 33 project commits, including their authors and messages.
It uses `git-filter-repo --subdirectory-filter projects/finetuning-labs` on a separate clone.
The extracted files matched the source Git tree exactly before migration changes.
The original repository and its history remain unchanged.

Commit IDs change because the project now starts at the repository root.
[The commit map](migration-commit-map.tsv) maps original IDs to extracted IDs.
Use the extracted ID for Git operations in this repository, including
`scripts/backfill_english_quality.py --baseline-commit`.
Historical reports retain original commit IDs, paths, URLs and exact file contents.
Those records still refer to the original repository.

The standalone checkout is `/home/jshah/Documents/GitHub/finetuning-labs`.
The original project folder remains in `jadoo-labs` as a backup.
Make new project changes in the standalone repository, not both copies.

## Changes for the standalone checkout

Evaluation provenance now reads the project repository instead of its old monorepo parent.
The recovery script reads `data/rewriter/current` directly from the extracted Git history.
Two tests exercise these operations against temporary Git repositories.

`make test` now requests the data and hosted dependency groups that the existing tests need.
The README and generated dataset-card source point to the new repository.
Colab needs GitHub access while this repository is private. Downloading and uploading the
notebook to Colab is an alternative.

The repository ignores `.env` and `.env.*`, except `.env.example`.
No credentials, virtual environments, personal data or ignored artifacts were copied.
No Hugging Face datasets, model repositories or hosted services changed during migration.

## Verification

The standalone checkout passed these commands:

```sh
make check
make build
make review export
```

The test suite passed all 196 tests. Ruff checks and the benchmark audit passed.
The build produced a source distribution and a wheel. Review and export commands produced
local outputs for all 128 benchmark cases. Five existing Pydantic deprecation warnings remain.
These checks do not establish model quality or replace human review.

Gitleaks 8.30.1 scanned the extracted history before publication. Its 20 findings were
checked against the corresponding historical JSON lines. All were file checksums or the
pinned Arka Git revision. No credential was confirmed. This scan is not a guarantee that
all material is suitable for public release. Review the history and data before changing visibility.
