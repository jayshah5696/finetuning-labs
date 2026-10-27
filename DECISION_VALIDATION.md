# Data sources and Luna Decisions validation

## Where the data came from

The training pilot is synthetic. The planner defines fictional situations, names, counts, quotes,
and filenames. GPT-6 Luna generates canonical documents and noisy variants. Long documents use
controlled errors added to the canonical text. The retained release has 219 pairs from 55 source
documents. It does not use a downloaded text corpus or personal transcription pairs.

The separate 68-case development benchmark contains 30 TypeWhisper authored fixtures and 38 new
authored cases. The selected 10 personal history entries remain local, pending review. They were
not sent to Luna, Sol, or the Decisions API and are not in the shared training release.

## How Arka is used

The first generation and Sol review used Arka's `LLMConfig` and `build_openai_client` factory at
commit `779040204b0f392465b7e6240bf9d9200816dc5d`. The rewriter project supplied the prompts,
saved calls, deterministic checks, and exports. It did not use Arka's decision stages. Sol was
kept as the first evaluator to match the diagnostic baseline. That choice did not require omitting
the decision client; the pinned Arka commit already supports it.

The new audit uses `arka.decisions.client.DecisionClient`, `DecisionConfig`, and `YesNoQuestion`.
It sends five named questions in one request to OpenRouter's alpha Decisions endpoint, requesting
`openai/gpt-6-luna-decisions`. It uses the clean pinned dependency, without modifying the local
Arka checkout. These are native decision calls, not JSON verdicts generated through Responses.

See [Arka's decision documentation](https://github.com/jayshah5696/arka/blob/779040204b0f392465b7e6240bf9d9200816dc5d/docs/decision-models.md),
the [OpenRouter model page](https://openrouter.ai/openai/gpt-6-luna-decisions), and the
[official OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions).
OpenAI's direct API and OpenRouter use different wire formats. This project uses Arka's
OpenRouter adapter and the existing OpenRouter credential.

## Criteria and policy

The questions check meaning, terms and literals, mode, readability, and output discipline.
The remote state contains only the original input, candidate rewrite, mode instructions, language,
and necessary profile or dictionary context. No reference answer, expected fixture label, source
identity, or earlier Sol verdict is sent. This removes the reference-equals-candidate setup of
the first training review.

The provisional policy passes only if every criterion has probability at least 0.95. A criterion
below 0.20 causes a failure flag. Other results are uncertain. These thresholds are starting
policy values, not measured probabilities of agreement with humans. No average can cancel a
failing criterion. Missing or invalid responses cannot pass. The caller rejects oversized evidence
before Arka can truncate it.

Each call is saved before spending and reused on resume. Exact requests, typed answers, and raw
HTTP request/response bodies are retained without headers or credentials. There is no fallback
to a different model or to chat generation. Changed question wording requires a new run path.

## Measured results

Both full runs checked 156 authored control outputs and all 219 training-pilot pairs. The controls
contain 80 authored acceptable outputs and 76 deliberately wrong outputs. Forty-nine wrong
outputs pass the declared mechanical checks and therefore test semantic review. These labels
are authored expectations, not independent human judgments. They were used during question
development, so these results are not an unseen accuracy estimate.

| Result | First wording | Clarified wording |
| --- | ---: | ---: |
| Authored acceptable outputs accepted | 62 of 80 | 71 of 80 |
| Authored acceptable outputs flagged as failures | 0 | 0 |
| Deliberately wrong outputs flagged as failures | 65 of 76 | 58 of 76 |
| Deliberately wrong outputs wrongly accepted | 0 | 1 |
| Deliberately wrong outputs left uncertain | 11 | 17 |
| Training pairs accepted by this policy | 113 | 157 |
| Training pairs left uncertain | 76 | 48 |
| Training pairs flagged as failures | 30 | 14 |

The first wording flagged 15 unchanged input/output pairs as failures. The clarified wording
states that identical text preserves meaning and literals, and that prompt mode produces a
request for later execution. It still flagged six unchanged prompt pairs on mode or other
criteria. These flags cannot be treated as confirmed data errors.

The clarified run also approved `slack-06-attack`: it changed a request to move a meeting into
an assertion that the meeting had moved. Its meaning probability was 0.96. That is a concrete
false approval and shows why a high returned probability is not proof of correct validation.

The second run has 62 training pairs in `review-queue.csv`. The published dataset and original
Sol results are unchanged. Use these scores to prioritize review. Do not automatically remove
the flagged pairs or certify the accepted pairs. Blind Sol adjudication and independent human
labels are needed to establish which disagreements are genuine errors.

## Reproduce

Run from the project directory with `OPENROUTER_API_KEY` available:

```sh
uv run --locked --group hosted python -m finetuning_labs.rewriter.decision_validation \
  --run artifacts/my-decision-audit --scope all
```

The current code uses the clarified questions. Run settings are in
`configs/decision-validation.toml`. Saved first and second question definitions remain in their
respective `run.json` and call files under `data/rewriter/decision-audits/`. Reusing an old run
with changed questions is rejected.

The next validation improvement is an explicit request-versus-assertion check, followed by blind
adjudication of disagreements. Evaluate any revised questions on fresh control families before
using them as an automatic training-data gate. Keep the shared 68-case baseline unchanged.
