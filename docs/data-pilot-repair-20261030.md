# English data pilot repairs, 30 October

> **Historical note:** This document records the 30-31 October 2026 bounded pilot repair runs across the three data expansion workstreams. See [status.md](status.md) for subsequent releases and evaluations.

The authoring briefs were repaired and fresh bounded pilots ran across the three workstream
threads starting at 14:26 UTC. Their exact run IDs and request times are in
`data/rewriter/workstreams.json`. The bounded repairs are complete. The coordinator verified
the result on 31 October. All three complete pilot gates failed, so scaling remains blocked.

## Fresh pilot results

| Stream | Fresh sources | Qualified training pairs | Short | Long | Missing pilot slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformation rules | 42 | 21 | 19 | 2 | 9 |
| Personal English patterns | 21 | 5 | 5 | 0 | 7 |
| Long coverage | 67 | 17 | 16 | 1 | 23 |
| Total | 130 | 43 | 40 | 3 | 39 |

There were 59 model-passing candidates before inspection. Inspection excluded another 16.
No fresh validation row qualified. Two accepted long pairs test corrections and dictionary
normalization; the third is an unchanged-input case. These are model and agent judgments,
not human gold labels. They do not establish readiness for another training run.

The rule stream now covers both correction directions and all three R07 dictionary/literal
pilot slots. Long coverage remains the main gap. Failed sources include repetition, editing
challenges confined to the opening, changed scope or certainty, and copied formatter headers.
Some long inputs were too short; the larger word budget also produced overlength Slack threads.
No acceptance range was relaxed and no text was truncated.

Arka's untrusted seed wrapper exposed an authoring boundary problem. The rule and long streams
amended only unused sources to move operational authoring requirements outside that wrapper.
This fixed input-header failures in the long stream's second round, but some targets still
copied context headers. The personal run had already completed its calls; its remaining
authoring-boundary risk is recorded, not claimed as fixed.

The personal short canary received a suspected false native fact veto. Its exact native state
and six cached answers bind correctly. There is no established transport mismatch. The veto
remains binding; no completed verdict was rerolled. Native fact calibration remains unresolved.

## Coordinator verification and exclusion

The coordinator read the actual viewer/SFT exports, verified their schemas and exact assistant
targets, and checked text overlap across all six first/fresh training exports. There were zero
matches against the existing release, benchmark fixtures and local personal holdouts, and zero
matches against earlier staged streams under the existing text heuristic. The original new
validation row also had zero matches against staged training. Semantic source-family and
privacy clearance remain pending. A text heuristic is not an independent semantic review.

The six exports contain 83 staged training rows before coordinator inspection. One older
long-pilot prompt target begins with MODE, LANGUAGE and INPUT headers. The coordinator excluded
that exact row using `data/rewriter/expansions/coordinator/quarantine.json`. Preserve the row and
its passing reviews as evidence. It is absent from current train, dev and validation data.
The remaining 82 staged candidates are not an approved combined release. All 43 fresh rows
are outside this one-row quarantine.

The coordinator verified 326 first-pilot manifest entries and 474 fresh-pilot entries without
hash mismatches. The rule exporter was replayed without hosted calls, followed by all 178
tests, lint, format checks, the benchmark audit and package builds. Personal and long branches
record 177 and 184 passing project tests and passing exports/builds. The rule runtime hit a
usage limit after committing; its artifacts were recovered and verified directly.
The main checkout also passed `just check` and `just build` (via [just](https://github.com/casey/just)) on 31 October, with 191 tests.
That checkout includes concurrent uncommitted expansion work; these checks do not establish
that the three data branches have been combined or that their release gates pass.

Each source branch is pushed. The data commits are `5f97dd6`, `2e84cf6` and `89bf3f4`.
The rule branch also includes documentation-only dependency commit `967136f`. Exact revisions,
thread links, handoffs and row counts are in `data/rewriter/workstreams.json`. Scope checks found
no changes to the active release, frozen holdouts or supported notebook. Credential-pattern
checks found no matches outside a typed opaque encrypted reasoning payload. That encrypted
payload was preserved unchanged and was not treated as an exposed credential.

No training or Hugging Face publication used these repair exports. The separate data-prep
thread `3ee0f519-0231-4ea5-945a-352578283295` has no final verified handoff in its latest parent
message. Its provisional root-checkout files are outside this result and remain untouched.

## Evidence from the first pilots

The first pilots staged 40 training pairs and one separate long validation pair. All three
failed their complete coverage or quality gates. Scaling stayed blocked. Thirteen rule-stream
rows were excluded on inspection after model passes. Other failures included short long-form
inputs, omitted constraints, unauthorized transcription edits and a retained canceled action.
These failures show why an exact model pass is insufficient for corpus quality.

The coordinator read back all four staged viewer/SFT exports. Counts, column sets and exact
assistant targets matched. Normalized input and target duplicates among the 40 training rows
were zero. Branch comparisons showed no changes to the current release, holdout freeze or
supported notebook.

The coordinator also ran the existing `boost_data.protected_index()` and `training.LeakageIndex`
from the main checkout. Its protected index included current train/dev/validation data, benchmark
fixtures and the local ten-record personal calibration set. The first-pilot viewer file hashes
bind this audit to the exact files below.

| Stream | Training rows | Viewer SHA256 | Protected-text matches | Earlier-stream matches |
| --- | ---: | --- | ---: | ---: |
| Transformation rules | 14 | a26047ba33404b3d79fa13dc1122b9cc9204eaea8a3d0ec4d4a63d4b0299b4a1 | 0 | 0 |
| Personal patterns | 7 | 5aa5f8ef84ac0c92f1cfd0043277b0992dfb93161bf002e9e1d96035f530e053 | 0 | 0 |
| Long coverage | 19 | c4b13b72baa40278682b8bb16e3df948728bce9485b3c064af6abec1a679d698 | 0 | 0 |

The one new validation row had zero text matches against those training rows. These heuristic
checks do not establish semantic independence, privacy clearance, source-family separation or
second-pilot readiness. Combined review and release verification remain pending.

## Repair requirements

- Preserve first-pilot files and completed verdicts. Fresh episodes need new source IDs and
  a separately frozen allocation. A new internal attempt is not a new public dataset version.
- Replace repeated authoring labels, quotes, code histories and closing exceptions with varied,
  scenario-specific facts. Do not make every document follow the same structure.
- Aim long inputs at 1,200 to 1,600 pinned Liquid tokens within the unchanged acceptance range
  of 1,000 to 2,000. Check author token limits, both student tokenizers, full-chat fit and the
  2,048-token generation reserve. Do not pad or truncate.
- Define applicable constraints before target authorship. Verify corrections, canceled actions,
  surviving requests, negation, conditions, uncertainty, literals, actors and source boundaries.
  Sol verifies exact pairs and does not write or repair targets.
- Run a small representative canary before each complete pilot. Allow at most two independent
  fresh source attempts per slot in this repair run. Never repeat a completed verdict to get a pass.
- Preserve all native vetoes and failed or uncertain Sol results. Inspect naturalness and
  repetition as well as per-row correctness before allowing scale.
- Keep direct personal records local. Use only independently fictionalized patterns for hosted
  generation and public staging. Preserve original calibration holdouts and unknown history modes.

The rule stream targets a fresh 30-slot pilot. The personal-pattern stream targets 12 slots.
The long stream targets 40 slots with separate validation and training scenarios. Existing
larger targets remain conditional on a complete passing pilot and corpus inspection.

Arka upstream HEAD was rechecked and still matches the shared pin
`5d353740ae1dd34742841071752814028db1489a`. No shared pin or training configuration changed.
Credential values must never appear in discovery commands, output, source or logs. Workers
must stop hosted work on a credential failure instead of searching for another key.

Nothing in this repair launch publishes data or starts student training. Integrate reviewed
results sequentially and verify cross-stream evidence before changing the current release.
