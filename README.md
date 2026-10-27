# Finetuning Labs

The user accepted the 68 cases as development validation on 27 October 2026. The acceptance and
immutable file hashes are in `data/rewriter/validation-freeze.json`. These cases, their output
fixtures, and their source families are excluded from training. Existing failures stay recorded.
This acceptance does not establish independent human judgments or an unseen final test.

The rewriter benchmark bundle has 68 calibration cases. It retains the original 60 short cases
and appends eight long examples, with two per mode. There are 30 imported TypeWhisper authored
cases and 38 new authored examples. The long examples come from four fictional source documents,
each adapted for two modes. They are eight cases from four source families, not eight independent
documents.

The benchmark has four columns: input, expected output, mode, and checks. Supporting records keep
case IDs, profile and dictionary context, literal checks, source identity, and review status.
The stock Liquid and Luna diagnostic runs and their shared Sol assessments are complete.
All human labels remain pending. A separate local bundle has
10 selected personal transcriptions with draft corrections. It is ignored by Git and excluded
from package builds. Neither bundle is a locked test or training dataset.

## Read and check the draft

Run commands from this directory:

```sh
uv sync --locked
make check
make review
make export
make build
```

`make review` writes `artifacts/benchmark-review.html`. Open it to inspect the four-column table.
Each case also has expandable acceptable and deliberately wrong outputs. All labels on those
examples are authored expectations. They are not independent human or model judgments.

`make export` writes `artifacts/benchmark-guide.jsonl`. Its only fields are `system`, `user`, and
`assistant`, following the supplied HTML guide. Profile, dictionary, and language context go into
the actual model request. Review scores, source records, and expected checks do not. This export
contains evaluation references and must not be used as training data. Export refuses to overwrite
an existing file; choose a new output path for another copy.

For explicit paths:

```sh
uv run --locked finetuning-labs audit --report artifacts/benchmark-audit.json
uv run --locked finetuning-labs review --output artifacts/benchmark-review.html
uv run --locked finetuning-labs export --output artifacts/another-evaluation-copy.jsonl
```

The package includes the draft data, so its audit also works from an installed wheel without the
TypeWhisper checkout. No default command loads weights, reads credentials, or calls a provider.

## Review expectations

Read [VALIDATION.md](VALIDATION.md) for failure cases and the acceptance rules. The reference is
one acceptable rewrite. Different valid wording can pass. A wrong fact cannot be offset by better
grammar or style.

Fill `data/rewriter/benchmark-draft/review/human-labels.csv` with `pass`, `fail`, or `uncertain`
and a short critique of each proposed reference. Leave unreviewed rows pending. This file is a
review worksheet; the offline audit does not treat it as model-quality evidence.

The bundle contains 156 authored output fixtures. Some wrong outputs intentionally pass literal
checks. The offline audit checks mechanical expectations only. GPT-6.1 Sol calibration and human
review are separate work. The draft judge instructions are in
`data/rewriter/benchmark-draft/supporting/judge-system.txt`.

## Long examples

The appended inputs contain 1,094 to 1,865 tokens with the pinned `LiquidAI/LFM2.5-230M`
tokenizer. Every mode has one input near 1,100 tokens and one near 1,850 tokens. The examples
cover full paragraphs, numbered sections, direct quotes, and conditions stated far apart.
Two source documents also use Markdown block quotes, and one includes indented bullets.
The original short cases, their rules, and their output fixtures are unchanged.

The new output fixtures include dropped final sections, changed quotes, reversed amount roles,
invented approval, and obedience to quoted instructions. Thirteen of the 16 new wrong outputs
pass the mechanical checks and need semantic review. These are authored failure expectations,
not measured judge results. All eight references still need human review.

Run the tokenizer-only length check explicitly:

```sh
make tokens
```

This optional command installs the locked `data` dependency group and downloads the pinned
tokenizer and chat template. It loads no model weights and calls no inference provider. The
normal offline check does not need those dependencies. It measures raw input, reference output,
the full inference prompt, and the complete reference chat without truncation. Chat controls and
system instructions count toward the context limit, as described in the
[Transformers chat-template guide](https://huggingface.co/docs/transformers/v5.21.0/chat_templating).

The longest complete reference chat is 3,877 tokens. Reserving 2,048 generated tokens needs at
most 4,065 tokens including the prompt. A 4,096-token limit fits this draft. The earlier
1,024-token setting cannot support these examples. These measurements establish text length;
Colab L4 memory use and model quality remain unmeasured. Recheck lengths after changing prompts
or data. The check exits with an error for long inputs outside 1,000 to 2,000 tokens or for a
context overflow. It never silently truncates text.

The saved evidence is `data/rewriter/benchmark-draft/supporting/long-form-tokens.json`.
The full 68-case report is generated at `artifacts/benchmark-tokens.json`. The review page links
directly to all eight long cases, with expandable full input and expected output. The four-field
data format is unchanged. These calibration references remain separate from a future training pool.

## Data and source boundaries

The imported snapshot is exact. Its hash, source revision, original IDs, and context are retained
in supporting files. TypeWhisper's repository license is copied to
`supporting/TYPEWHISPER-LICENSE.txt`; no separate fixture license was verified. The imported cases
are not being relicensed as Apache-2.0.

All cases and their variants belong to calibration. Do not use those source families in a locked
test. The existing TypeWhisper model outputs are not target answers; their human review is pending.
The user authorized source selection with "Use anything." Ten raw history entries now have
proposed corrections under the ignored `data/rewriter/personal/calibration-10/` directory. The
saved historical app outputs are retained as evidence, not preferred corrections. Review these
private cases locally using the commands in that directory's README. They are not uploaded to
Sangam or included in the shared bundle.

## Later comparison

The selected student is `LiquidAI/LFM2.5-230M`. GPT-6 Luna is the cloud baseline and teacher.
GPT-6.1 Sol is the shared semantic evaluator. Training and paired student inference target
Google Colab L4. The diagnostic runner now supports Liquid on a Modal L4 and Luna/Sol through
OpenRouter with Arka's client factory. The Colab baseline notebook uses the same runner and is
syntax-checked, but has not been executed in Colab. The training path remains pending.

After reviewing the draft and checking the evaluator, prepare independent development and test
cases. Freeze their source groups, then run the baseline before creating the training pool.

## Authorized publication and diagnostic evaluation

The user authorized a private Hugging Face upload, local Git commits, and actual Liquid/Luna
baseline runs. See [scripts/README.md](scripts/README.md) for exact commands and resume behavior.
Settings are in `configs/rewriter.toml`. The historical long-example authoring script is retained.
The new runner writes per-case requests and results before producing metrics. It never treats
missing or failed calls as passes. Sol results remain advisory until human calibration.

The API provider is OpenRouter. Use the existing `OPENROUTER_API_KEY` environment variable.
Keys never enter run metadata, dataset files, or Git. No personal transcriptions enter these runs.

Open `notebooks/rewriter_baseline_l4.ipynb` in Colab for the portable baseline. It downloads the
wheel and config from the private dataset and verifies their hashes. It retains Colab's CUDA
PyTorch and records its actual version. Use the same environment for the later paired comparison.

The private dataset is [jayshah5696/finetuning-labs-rewriter](https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter).
Its 68 rows and supporting files were downloaded and hash-verified after publication. Colab runtime
assets and the collected Liquid outputs are also published with immutable revision receipts under
`data/rewriter/releases/`. The private Dataset Viewer returns 501 on this account because it needs
PRO or Enterprise access. Authenticated file downloads work; the dataset remains private.

The first Liquid run used an NVIDIA L4 on Modal and produced 68 end-token completions with no
output-limit truncations. Only 53 of 68 outputs passed declared mechanical checks. All eight long
cases failed a check, and none of the three explicit identity cases stayed unchanged. This is
not a semantic pass rate. The per-case outputs and [mechanical report](data/rewriter/evals/baseline-20261027/liquid-mechanical/REPORT.md)
are retained for review. The runtime used reference convolution kernels, so its timings are not
an optimized deployment benchmark.

The restored zshrc credential authenticated through OpenRouter. It was read into process memory;
the file, its symlink, and its target were left unchanged. Luna generated all 68 cases and Sol
assessed all 136 candidates with the same rubric and settings. There were no generation-limit
truncations or judge errors.

Sol-assessed whole-case passes were 2/68 for stock Liquid and 62/68 for Luna. The headline that
weights all four modes equally was 4.2% versus 92.7%. Long-case passes were 0/8 versus 6/8, and
explicit identity preservation was 0/3 versus 3/3. These are diagnostic model judgments, not
human-calibrated accuracy. The earlier 53/68 Liquid number measured mechanical checks only.

The [paired report](data/rewriter/evals/baseline-20261027/comparison/REPORT.md) is recomputed from
exact saved requests and responses, with candidate hashes and matching judge settings checked.
The [review priorities](data/rewriter/evals/baseline-20261027/comparison/REVIEW.md) cover Luna's
six failures and judge-label questions. All new hosted artifacts were downloaded and verified at
Hub revision `f740b9d7908890f5317416c53ce27b174b55bdd9`. No fine-tuning has run.
