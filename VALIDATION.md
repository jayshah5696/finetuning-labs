# Validation contract

This draft defines what a rewriter should do. It does not measure a model yet.

## What can fail

- A row has missing text, an unknown mode, or extra metadata mixed into the visible benchmark.
- A source case is duplicated, loses its profile or dictionary context, or changes after import.
- A reference contradicts its own literal checks.
- A substring check accepts `Rinaldo` when the required name is `Rina`.
- A number check rejects a declared equivalent such as `three` and `3`.
- A literal check accepts changed spelling or case in code and filenames.
- A clean input changes even though this case explicitly requires an unchanged result.
- A wrong answer retains all required terms and receives a false semantic pass.
- An output drops negation, reverses a correction, adds a requirement, or obeys dictated instructions.
- A valid alternative rewrite fails because it differs from the reference wording.
- A judge result has missing criteria, the wrong case, contradictory verdicts, or no critique.
- An uncertain judgment or missing judge result is counted as a pass.
- HTML review exposes markup from a case as executable content.
- An export omits required context or overwrites an earlier artifact.
- A long input is called 1,000 tokens based on word count rather than the selected tokenizer.
- A length check omits chat controls, system instructions, or space for the generated output.
- A long rewrite stops early, drops a numbered section, changes a quote, or resolves a condition
  using evidence that applies to a different decision.

## Checks we can run without a model

Validate the four benchmark fields, modes, supporting identities, source snapshot hash, and
fixture references. Check declared required terms, forbidden terms, exact literals, and identity
cases. Match terms as words rather than arbitrary substrings. Accept only explicitly listed
number equivalents. These checks can reject a candidate but cannot approve its meaning.

## Semantic checks

Use the same rubric for every candidate. Check meaning, terms and literals, requested mode,
readability, and output discipline. Return `pass`, `fail`, or `uncertain` for each criterion,
with a short critique and an explicit critical-error list.

A known critical error fails the case. Every applicable criterion must pass for an overall pass.
Otherwise, unresolved meaning remains uncertain. Judge errors leave the assessment pending.
An absent semantic judgment remains pending even when every deterministic check passes.

The reference is one acceptable output. Accept equivalent wording unless a case explicitly
requires unchanged text or exact literals. Judge against the input and mode, not the model name.
Never reward length or formality by itself.

The semantic fixtures include errors that deliberately pass literal checks. Their authored labels
must never be reported as measured judge performance. GPT-6.1 Sol calibration and human review
are still pending. No automatic judge is called by `make check`, `make review`, or `make export`.

## Report outcomes

Keep passes, failures, uncertain assessments, judge errors, and unassessed cases separate.
Report generation failures and truncations separately. Show counts beside percentages.

A whole-case pass requires a complete generation, every mechanical check, and all five semantic
criteria to pass. Changed meaning cannot be offset by better grammar. Equivalent wording can
pass when the mode allows it. Uncertain or missing judgments never establish success.

For a frozen run, `N` is every scheduled case, `P` is verified passes, and `F` is definite quality
failures, including mechanical failures. Keep `N` fixed when a generation or judge request fails.
Report each case once, with any retry history. Do not select the best of repeated generations.

| Metric | Definition and use |
| --- | --- |
| Verified pass rate per mode | `100 * P_mode / N_mode`. Main quality result. Every scheduled case stays in the denominator. |
| Headline score | Mean of the four mode pass rates. Give each mode equal weight because this draft has more transcribe cases. |
| Pooled verified pass rate | `100 * P / N`. Show the total successful cases alongside the headline. |
| Decided-case pass rate | `100 * P / (P + F)`. Secondary diagnostic, always beside its denominator and coverage. |
| Assessment coverage | `100 * (P + F) / N`. Show how much has a definite quality verdict. |
| Critical-error rate | Outputs with a confirmed meaning error divided by `N`. Show semantic review coverage too; unreviewed output is unknown. |
| Identity rate | Unchanged successful outputs divided by scheduled explicit identity cases. |
| Execution and assessment failures | Separate truncation, generation-error, judge-error, and uncertain counts, each divided by `N`. |

Each case has one final status: pass, fail, uncertain, judge error, generation error, truncated,
or pending. Generation failures and truncations take precedence over quality scoring. A known
mechanical failure is enough for fail. Otherwise, an absent or invalid judge result remains
pending or judge error. A zero denominator produces `N/A`, not zero percent.

Show short and long cases separately. Show failure counts for meaning, literals, requested mode,
readability, and output discipline. Categories can overlap. A candidate that retains both amounts
but swaps which one is the deposit fails meaning even when every literal check passes.

Training loss and text similarity are diagnostics. They do not establish whether a quote,
negation, or condition survived. Do not use them as the main acceptance metric.

## Validate the judge

GPT-6.1 Sol is the configured semantic judge. Use the same rubric, settings, and context for
every model, and hide generator identity. Save the exact candidate, requested and returned model,
rubric hash, and critique. Rescore all comparison models if the rubric or judge changes.

The 156 fixtures contain 80 expected acceptable and 76 expected wrong outputs. These authored
labels need human confirmation. The immediate next action is a 40-output pilot, ten per mode,
with five acceptable and five wrong candidates per mode. Include all eight long references and
all 16 long wrong outputs. Review the complete texts without showing proposed labels first.
The selected review batch and model runner are not implemented yet.

Compare Sol with definite human labels using these measures:

| Judge metric | Definition |
| --- | --- |
| False acceptance | Human-fail candidates the judge passes divided by human-fail candidates. |
| Valid-output acceptance | Human-pass candidates the judge passes divided by human-pass candidates. |
| Human agreement | Matching definite verdicts divided by all human-labeled pass/fail candidates. Judge errors and uncertainty are disagreements. |
| Decision coverage | Definite judge pass/fail verdicts divided by all human-labeled pass/fail candidates. |

Show human-label coverage and raw counts. Proposed pilot gates are zero accepted critical errors,
at least 95% valid-output acceptance, and complete decision coverage. For 20 human-confirmed good
outputs, 95% means at least 19 accepted. This is a small setup check, not proof about unseen data.
Review disagreements. After tuning the judge prompt, check fresh human-labeled cases that were
not used to tune it. Do not relabel a human disagreement simply to match Sol.

Task-specific checks and comparison with human judgments follow
[OpenAI's evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
The formulas and proposed thresholds here are project choices, not universal standards.

## Compare the models

After the judge is checked, prepare an independent test set. Freeze its cases before training.
Keep related source families together and out of the training pool. The current 68 cases are
for judge setup and diagnosis, so they cannot establish final generalization quality.

Run stock and tuned Liquid with the same tokenizer, prompt, decoding, context limit, generation
allowance, and frozen test. The adapter is the primary changed variable. Give Luna the same
behavior contract and record its actual API settings separately. Judge all candidates identically.

Report paired results: both pass, only tuned passes, only stock passes, and neither passes.
Pooled improvement in percentage points is `100 * (tuned_only - stock_only) / N`. Show per-mode
and short/long changes too, with unknown outcomes and the number of source families visible.

Existing proposed model targets are zero observed critical errors after full review, at least
90% verified passes in every mode, and at least 95% identity. Require complete assessment coverage
before accepting a run. These are targets, not measured results or a guarantee. The shared draft
has only three explicit identity cases, so all three must stay unchanged to reach 95%. Its eight
long cases represent only four source documents. Report counts to expose those limits.

Record latency, peak memory, and model size separately. Colab L4 timing does not establish desktop
timing. Deployment measurements need the actual desktop runtime and export.

## Current evidence and stopping point

Data and token-length checks are implemented. The 68 references, 156 fixture checks, and 22
software tests pass their offline checks. Full reference chats fit 4,096 tokens. These results
establish data integrity and declared mechanical expectations, not model accuracy.

No Luna, stock Liquid, tuned Liquid, or Sol judge output has been measured. Human labels are
pending. This contract defines the future score report; aggregation and model runners are not
implemented. Stop here before model calls or training. Next comes the human-review and judge
pilot, then independent test data and the Colab L4 stock baseline before training-data preparation.

## Draft boundary

All existing TypeWhisper cases and their variants belong to calibration. New authored cases in
this draft also belong to calibration. This is not the locked product test or a training pool.
Ten selected personal transcriptions stay in a separate local bundle. Their proposed corrections
remain pending human review and are excluded from shared publications and package builds.

## Long documents

The eight appended examples represent four authored source families. Keep each family's mode
variants together. Their source groups are recorded in the supporting case records. They are
calibration examples and must not be split into training and locked-test sets.

Run `make tokens` to measure with the pinned Liquid tokenizer and native chat template. Require
1,000 to 2,000 tokens for each designated long input. Count the complete reference chat and the
prompt plus the generation allowance separately. Check both against the selected context limit.
Any overflow is an error. Do not cut a long case down to fit without a new review.

Semantic review must read the complete input and output. Check the last paragraph as carefully
as the first one. Check every numbered section and the meaning of each quote in context. Verify
the relation between a condition and its decision even when the evidence occurs paragraphs apart.
Keeping a number or phrase somewhere in the output does not prove that its role is preserved.
