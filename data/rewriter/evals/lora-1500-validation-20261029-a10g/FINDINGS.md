# Updated 230M LoRA findings

The 1,500-example adapter improves the frozen diagnostic from 48/84 to 53/84 passes, a gain
of 6.0 percentage points. It gains six cases and loses one previously passing case. The gain
is modest. The adapter still misses the project's quality targets and should not be promoted.

The checked adapter is `ba71856780e4d2398218fe76756b3390556437ea`. All 164 weight tensors differ
from the previous adapter and are finite. Its card records the correct 1,500-row dataset commit
and validation loss 0.311061. Lower loss does not establish rewrite accuracy.

| Slice | Stock | Previous 1,000-example LoRA | Updated 1,500-example LoRA |
| --- | ---: | ---: | ---: |
| Slack | 0/16 | 14/16 | 15/16 |
| Transcribe | 0/35 | 19/35 | 21/35 |
| Prompt | 4/16 | 6/16 | 7/16 |
| Polish | 0/17 | 9/17 | 10/17 |
| Overall | 4/84 | 48/84 | 53/84 |
| Equal-mode score | 6.25% | 58.1% | 64.1% |
| Long cases | 0/13 | 3/13 | 4/13 |
| Identity preserved | 0/8 | 8/8 | 7/8 |
| Judge-flagged critical errors | 33 | 25 | 18 |
| Truncated outputs | 0 | 0 | 1 |

Critical-error coverage differs: the previous adapter has 84 semantic reviews; the updated
adapter has 83 because one output truncated. That output remains a failure in the 84-case
denominator, and its absent semantic review does not establish safety. On the same 83 complete
cases, critical errors decrease from 24 to 18. There are no judge errors or uncertain verdicts.

## Gains and regressions

Six cases newly pass: `tw-fillers-04`, `tw-facts-05`, `prompt-07`, `polish-06`, `long-slack-02`
and `long-polish-01`. The updated model preserves the findings-only constraint in `prompt-07`,
keeps the unfinished conditional in `tw-facts-05`, and handles the explicit timeout correction
in `polish-06`. Forty-seven cases pass with both adapters; 30 pass with neither.

The previously passing `arka-v2-polish-identity-long` now fails the exact unchanged-input rule.
Its edits preserve meaning but are unnecessary, including “were not clear” becoming “were
unclear.” The deterministic identity check correctly overrides the semantic judge's pass.

`arka-v2-prompt-24` now reaches the unchanged 2,048-token generation limit. It starts repeating
material under another execution-summary and agenda section and ends mid-output. This is a
new truncation; the previous output completed but failed the semantic checks.

`arka-v2-polish-21` adds a new judge-flagged causal error by making sadness about leaving sound
like another reason for leaving. It already failed before, so this regression does not appear
as a lost whole-case pass.

Important targeted failures remain:

- `tw-fillers-01` now preserves “send” instead of changing it to “Say,” but still keeps “Um.”
- `arka-v2-transcribe-21` retains Thursday but drops the driving-lesson rescheduling request
  and its original Tuesday date. The correction is still mishandled.
- `arka-v2-prompt-21` now includes the supplied code, but invents a function wrapper and a
  duplicate indented snippet. It also drops beginner context and the ordering question.
- `prompt-09` still directs estimating missing inputs rather than making an estimate only
  when enough information exists, with a fallback to list missing inputs.

Meaning criteria passes increase from 60 to 65, literal passes from 71 to 74, and mode passes
from 49 to 58. Readability falls from 78 to 75 and output discipline from 83 to 82. These
criterion counts use all 84 scheduled cases; the unreviewed truncation cannot pass any criterion.

## Verification and interpretation

Both arms ran freshly on one NVIDIA A10 with the same base revision, tokenizer, prompts,
BF16 base weights, FP32 adapter tensors and greedy configuration. All 84 stock outputs match
the previous run exactly. The updated adapter changes 43 outputs; 41 remain byte-identical.
No validation or training data, notebook, rubric or generation budget was changed for this run.

Exact old Sol calls are reused for 84 stock outputs and 41 unchanged adapter outputs after
full packet/config/hash validation. Another 42 complete changed outputs receive fresh Sol
calls. All 167 eligible judgments are valid. Source paths and hashes are in judge-reuse.json.
This avoids new judgment variation on unchanged candidates. It does not supply human labels.

One calibration question remains visible: Sol accepts “Yeah” for spoken “uh huh” in
`tw-fillers-04`. A stricter spoken-word-preservation policy may reject that paraphrase. The
reported score retains the unchanged rubric and actual verdict; it was not adjusted after
seeing the result. Human calibration and an independent unseen test remain pending.

The next improvement should address the concrete correction, supplied-code, identity and
long-output failures before another general data expansion. This run does not establish which
training parameter or data change caused each behavior. The notebook and dataset remain unchanged.

The complete current report is comparison/REPORT.md. comparison/case-reviews.csv contains all
84 exact inputs, references, stock outputs, adapter outputs, statuses and critiques. The prior
comparison remains in the sibling lora-validation-20261028-a10g directory. before-after.json
records case-level transitions against that adapter. No desktop or L4 speed claim is made.
