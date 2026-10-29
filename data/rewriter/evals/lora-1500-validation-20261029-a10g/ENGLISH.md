# English-only results and critical errors

English can be the scope for the next training iteration. The existing dataset contains
1,364 English training pairs, 89 dev pairs and 81 validation pairs. This report selects
English from saved evaluation records. It does not change the dataset or notebook, remove
other languages, rerun a model, or change the frozen 84-case comparison.

| English slice | Previous LoRA | Updated LoRA |
| --- | ---: | ---: |
| Whole-case passes | 46/81 | 51/81 |
| Slack | 14/16 | 15/16 |
| Transcribe | 17/32 | 19/32 |
| Prompt | 6/16 | 7/16 |
| Polish | 9/17 | 10/17 |
| Exact identities | 8/8 | 7/8 |
| Truncated outputs | 0 | 1 |

The updated model passes 63.0% of English cases. Seventeen English outputs have judge-flagged
critical errors among 80 complete outputs. Another output truncated and remains a failure.
The flagged English outputs are six transcription, seven prompt and four polish cases.
Seventeen of the 18 critical-error outputs in the full comparison are English.

A critical error changes, invents or removes something that changes the user's intended
meaning or task. It can change a person, amount, condition, prohibition, code snippet, or
necessary detail. Count an output once even when its critical-error list has several entries.

| Actual English input | Actual updated output | Error |
| --- | --- | --- |
| send the draft to Rina not Ravi | send the draft to Ravi | Changes the recipient to the explicitly rejected person. |
| the invoice is $47.50 not $74.50 | The invoice is $74.50, not $74.50. | Changes the correct amount to the rejected amount. |
| make a python CLI use uv no network calls output csv | Write a Python CLI using `uv` to make network calls and output the results as a CSV file. | Reverses a prohibition into a requirement. |
| if the customer approves then we can | If the customer approves, we can proceed. | Invents a completion for a deliberately unfinished thought. |

The supplied-code case also invents a `def names():` wrapper, duplicates the original snippet,
and drops part of the explanation request. Long cases lose final assessments or restrictions
such as no email or cloud-storage connections. The exact lists for all 17 English outputs are
in english-only.json; complete inputs and outputs are in comparison/case-reviews.csv.

Grammar, punctuation, mode errors, identity violations and truncation are tracked separately.
An unnecessary meaning-preserving edit can fail identity without being a critical meaning
error. A truncated output is unreviewed for semantics, not confirmed free of critical errors.
These flags are Sol judgments, not independently confirmed human labels.
