# Reproduce the five-model English comparison

The shared source is the frozen English view of
`data/rewriter/reviewed-v2/validation-bundle`. All five arms use the same 81 case IDs in the
same order, with the same input, dictionary/profile context, reference answer, rules and Sol
rubric. The public validation split is pinned to dataset commit
`00d59dda95867fe5bcb147b1b1c5a34b5ad7f071`. No reference answer enters generation.

The existing stock/LoRA runs are collected from Modal. Their outputs are retained unchanged.
This update makes 81 fresh GPT-6 Luna generation calls through Arka and OpenRouter, then
reuses 49 exact matching Sol judgments and makes 32 new Sol calls. No completed verdict is
rerolled. A missing, failed, uncertain or truncated result never becomes a pass.

## Generate and grade

Supply the authorized OpenRouter credential through `OPENROUTER_API_KEY`. It is not stored
in this repository. Existing run directories resume saved calls without spending again.
Use a new run ID for an independent sample rather than overwriting an existing run.

```sh
uv run --locked --group hosted python -m finetuning_labs.rewriter.evaluation \
  --english-only --bundle data/rewriter/reviewed-v2/validation-bundle \
  generate --model luna \
  --output data/rewriter/evals/english-luna-validation-20261029/luna --workers 8

uv run --locked --group hosted python scripts/judge_english_models.py \
  --run data/rewriter/evals/english-230m-validation-20261029-a10g \
  --run data/rewriter/evals/english-350m-validation-20261029-a10g \
  --hosted-run data/rewriter/evals/english-luna-validation-20261029 \
  --output data/rewriter/evals/english-all-models-20261029 --workers 8
```

Liquid uses BF16 base weights, FP32 LoRA tensors, eager attention, seed 42 and greedy decoding
on Modal A10-family GPUs. The context limit is 4,096 and generation allowance is 2,048 tokens.
Hosted Luna uses `openai/gpt-6-luna`, reasoning effort `none`, a 2,048-token output budget,
and provider-default sampling. Sol uses `openai/gpt-6.1-sol`, reasoning effort `high`, the
unchanged rubric and a 16,384-token response allowance. Model labels are hidden from its packet.
Different model tokenizers and hosted sampling prevent a claim of identical inference mechanics.
This is a quality comparison, not an on-device latency or cost benchmark.

## Verify without API calls

The following reconstructs every assessment from hashed raw calls, checks the complete frozen
case set and recomputes every metric. It requires no hosted credential or GPU.

```sh
uv run --locked python - <<'PY'
import csv
import json
from pathlib import Path
from finetuning_labs.rewriter.comparison import _bound_assessments
from finetuning_labs.rewriter.evaluation import aggregate, evaluation_bundle

root = Path("data/rewriter/evals/english-all-models-20261029")
bundle = evaluation_bundle(
    Path("data/rewriter/reviewed-v2/validation-bundle"), english_only=True
)
report = json.loads((root / "comparison.json").read_text())
table = list(csv.DictReader((root / "case-reviews.csv").open()))
assert [r["case_id"] for r in table] == [c.id for c in bundle.cases]
assert len(table) == 81 and len(report["arms"]) == 5
for arm in report["arms"]:
    run = Path("data/rewriter/evals") / arm["run"]
    rows, records, config = _bound_assessments(
        bundle, run / arm["arm"], run / (arm["arm"] + "-judge")
    )
    assert config == report["judge_config"]
    assert aggregate(bundle, rows) == arm["metrics"]
    assert not arm["metrics"]["pooled"]["judge_error_events"]
print("All five frozen case sets and saved assessments verified.")
PY
```

`verification.json` additionally checks all 405 CSV outputs and assessments against their raw
sources, identical source/context/reference fields in every arm, exact cached-judgment bytes,
and consistent judgments whenever two models produce the same exact packet. It records 405
verified generations, 404 completed semantic reviews and one truncated student output.
The source manifest and assessment-file hashes are retained in that verification record.

The 230M adapter is still `ba71856780e4d2398218fe76756b3390556437ea`, trained on the earlier
1,500-row source. The 350M adapter is `b42778bcea7f4434365483e7b0ffd58e292fcead`, recording
1,864 English training rows. Both Hub heads were checked before evaluation. This does not
isolate model size from the training-data change. Historical results and judgment disagreements
remain preserved in their original run directories.

These are model judgments on development validation. Human judge calibration, an unseen test,
desktop latency and exported-model quality remain unverified.
