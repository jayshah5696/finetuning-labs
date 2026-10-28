# Trained 230M LoRA findings

The LoRA improves this diagnostic substantially, but it does not meet the project's quality
targets. Whole-case passes increased from 4/84 to 48/84, a gain of 52.4 percentage points.
The equal-mode score increased from 6.25% to 58.1%. All 168 generations and all 168 Sol
assessments completed. There were no truncations, judge errors or uncertain verdicts.

| Mode | Stock passes | LoRA passes |
| --- | ---: | ---: |
| Slack | 0/16 | 14/16 |
| Transcribe | 0/35 | 19/35 |
| Prompt | 4/16 | 6/16 |
| Polish | 0/17 | 9/17 |
| Long cases | 0/13 | 3/13 |
| Explicit unchanged-input cases | 0/8 | 8/8 |

## What improved

Output discipline improved from 8/84 to 83/84 semantic passes. The stock model often echoed
MODE, LANGUAGE and INPUT headers or added notes. The LoRA usually returned just the rewrite.
All eight explicit unchanged-input cases were preserved exactly and passed the full check.
There were 46 LoRA-only whole-case passes, two stock-only passes, two shared passes and
34 cases where neither model passed.

## What remains wrong

Meaning passes improved from 52/84 to 60/84. Term-and-literal passes decreased from 72/84
to 71/84. Judge-flagged critical errors decreased from 33 to 25. The large whole-case gain
therefore does not establish reliable factual or literal preservation.

Examples from the exact saved outputs:

- `tw-fillers-01` changes "send the report" to "Say the report".
- `arka-v2-transcribe-21` drops the corrected destination day, Thursday, from a Portuguese
  request to reschedule a driving lesson.
- `arka-v2-prompt-21` drops the supplied Python snippet and replaces the constraint to use
  that snippet with a request to provide code.
- `prompt-07` is a regression. "Return findings only" becomes a restriction to review race
  conditions only, losing the output constraint.
- `prompt-09` is a regression. The condition to estimate only with enough information and
  the fallback to list missing inputs become "Estimate the missing inputs for the estimate."

Only three long cases pass. The model still needs work on long-document completeness,
instructions and conditions, explicit spoken corrections, code and exact literal preservation.
It has not reached the targets of 90% passes in every mode and zero observed critical errors.

## Interpretation and next data work

The 230M model learned much of the output format. Meaning and exact-copy fidelity remain the
main limits. Add new, independent training examples for the failure categories above. Keep
all 84 validation cases frozen and outside training; do not turn these failed cases directly
into training examples. Retest with the same rubric and settings after any new training run.

These are Sol judgments on development validation, not human gold or an independent final test.
One known judge inconsistency is visible in `arka-v2-prompt-22`: Sol accepted stock metadata
headers as harmless even though the case requires unchanged text. The deterministic identity
check rejected it, so it did not count as a whole-case pass. The rubric was not changed and
no failed response was replaced with a more favorable judgment.

Both arms ran on the same NVIDIA A10 after the L4 attempt failed to allocate a worker.
Base weights used BF16 and LoRA tensors used FP32. No L4 or desktop speed claim is made.
The complete report is `comparison/REPORT.md`; `comparison/case-reviews.csv` contains all
84 inputs, references, model outputs, statuses, critical-error flags and critiques.
