# English stock, LoRA and Luna comparison

All arms use the same 81 English validation cases, prompts, 2,048-token output budget,
and Sol rubric. Liquid models use greedy decoding. Hosted Luna uses reasoning effort none
and provider-default sampling. Every scheduled case stays in the denominator.
These are model judgments on development validation. Human verification and an unseen test are pending.

| Model | Arm | Passes | Critical errors | Truncated | Identity unchanged | Long passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LiquidAI/LFM2.5-230M | stock | 4/81 (4.9%) | 33 | 0 | 0/8 | 0/13 |
| LiquidAI/LFM2.5-230M | lora | 50/81 (61.7%) | 17 | 1 | 7/8 | 4/13 |
| LiquidAI/LFM2.5-350M | stock | 18/81 (22.2%) | 50 | 0 | 2/8 | 0/13 |
| LiquidAI/LFM2.5-350M | lora | 62/81 (76.5%) | 13 | 0 | 7/8 | 7/13 |
| openai/gpt-6-luna | luna | 73/81 (90.1%) | 1 | 0 | 4/8 | 10/13 |

| Mode | LFM2.5-230M stock | LFM2.5-230M lora | LFM2.5-350M stock | LFM2.5-350M lora | gpt-6-luna luna |
| --- | ---: | ---: | ---: | ---: | ---: |
| /slack | 0/16 | 15/16 | 5/16 | 15/16 | 16/16 |
| /transcribe | 0/32 | 18/32 | 3/32 | 24/32 | 27/32 |
| /prompt | 4/16 | 7/16 | 4/16 | 11/16 | 15/16 |
| /polish | 0/17 | 10/17 | 6/17 | 12/17 | 15/17 |

Critical errors count outputs with changed, invented, or omitted meaning. Incomplete outputs
are reported separately and have no semantic verdict. An unchanged identity is an exact match.
The full configurations, adapter revisions, assessments and exact cached-judgment sources are saved alongside this report.

## Interpretation and limits

Luna leads the new 350M LoRA by 11 passes on the same 81 cases, or 13.6 percentage points.
Its eight failures include four exact unchanged-input violations, a retained canceled instruction,
two transcription-mode failures, and one changed chronology in a long transcription. One of the
mode judgments concerns punctuation after an unfinished thought and has a conflicting historical
judgment. That disagreement remains documented; it was not rerolled for a favorable result.

The 230M upload still records the older 1,500-row training source. Its previous 51/81 result
differs by one historical judgment from the current common-cache result of 50/81. Its weights
and output text are unchanged. The new 350M adapter records the 1,864-row English data.
Differences between the two adapters therefore combine model size and training data.

`case-reviews.csv` has exactly 81 source rows, each with all five complete output texts and
assessments. `verification.json` records readback checks against all 405 saved generations and
404 completed semantic reviews. The one truncated student output remains a failure and has no
semantic judgment. There are no judge errors or pending judgments.

The old 59/64 Luna result is a separate historical subset and is not the current headline.
The current Luna run generated fresh responses for the full 81-case set. Hosted sampling can
change outputs between runs. See RUN.md for exact settings, commands and offline verification.
