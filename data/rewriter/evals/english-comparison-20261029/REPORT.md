# English stock and LoRA comparison

All arms use the same 81 English validation cases, prompts, greedy decoding,
2,048-token output budget, and Sol rubric. Every scheduled case stays in the denominator.
These are model judgments on development validation. Human verification and an unseen test are pending.

| Model | Arm | Passes | Critical errors | Truncated | Identity unchanged | Long passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LiquidAI/LFM2.5-230M | stock | 4/81 (4.9%) | 33 | 0 | 0/8 | 0/13 |
| LiquidAI/LFM2.5-230M | lora | 50/81 (61.7%) | 17 | 1 | 7/8 | 4/13 |
| LiquidAI/LFM2.5-350M | stock | 18/81 (22.2%) | 50 | 0 | 2/8 | 0/13 |
| LiquidAI/LFM2.5-350M | lora | 62/81 (76.5%) | 13 | 0 | 7/8 | 7/13 |

| Mode | LFM2.5-230M stock | LFM2.5-230M lora | LFM2.5-350M stock | LFM2.5-350M lora |
| --- | ---: | ---: | ---: | ---: |
| /slack | 0/16 | 15/16 | 5/16 | 15/16 |
| /transcribe | 0/32 | 18/32 | 3/32 | 24/32 |
| /prompt | 4/16 | 7/16 | 4/16 | 11/16 |
| /polish | 0/17 | 10/17 | 6/17 | 12/17 |

Critical errors count outputs with changed, invented, or omitted meaning. Incomplete outputs
are reported separately and have no semantic verdict. An unchanged identity is an exact match.
The full configurations, adapter revisions, assessments and exact cached-judgment sources are saved alongside this report.

The available 230M adapter is the old 1,500-row training run at
`ba71856780e4d2398218fe76756b3390556437ea`. Its weights are unchanged. The new 350M adapter at
`b42778bcea7f4434365483e7b0ffd58e292fcead` records 1,864 English training rows. The comparison
therefore changes both model size and training data. It does not isolate the effect of size.

The 350M LoRA gains 44 passes over its own stock model, or 54.3 percentage points. It gains
12 passes over the available 230M LoRA, or 14.8 percentage points. It passes all 18 cases its
stock model passes and adds 44 more. Its 19 remaining failures include 13 critical-error outputs.
Both adapters preserve 7/8 exact identity cases. Long-case passes increase from 4/13 to 7/13.
The 350M LoRA passes 16 cases the 230M LoRA fails, but loses four previous passes. The regressions
are `arka-v2-polish-23`, `arka-v2-slack-24`, `tw-fillers-04` and `tw-repairs-02`. See
`adapter-transitions.json`. The gain is not uniform across cases.

The earlier 230M report showed 51/81. One existing exact-packet Sol judgment disagrees with
another about a period after an unfinished thought. The current common reuse policy keeps the
failure, giving 50/81. The weights and outputs did not change. The saved disagreement report
shows both judgments. This uncertainty is another reason to treat these scores as diagnostic.

## Checked examples from the 350M LoRA

| Input | Actual output | Result |
| --- | --- | --- |
| send the draft to Rina not Ravi | Send the draft to Rina, not Ravi. | Pass. The recipient is correct. |
| the invoice is $47.50 not $74.50 | The invoice is $47.50, not $74.50. | Pass. The amount is correct. |
| send 47 files I mean 74 files | Send 47 files. I mean 74 files. | Fail. The superseded instruction remains. |
| send the report tomorrow scratch that save it as a draft | Send the report tomorrow. Save it as a draft. | Fail. The canceled sending instruction remains. |
| if the customer approves then we can | If the customer approves, then we can proceed. | Fail. It invents a completion. |

The Python CLI case still reverses "no network calls" into a requirement to use network calls.
The longer meeting correction changes the final day back to Monday and drops the time.
These concrete failures need further work before using the model for unattended rewriting.
