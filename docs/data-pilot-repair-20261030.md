# English data pilot repairs, 30 October

The user approved repairing the authoring briefs and running fresh bounded pilots. The same
three workstream threads resumed at 14:26 UTC. Their exact run IDs and request times are in
`data/rewriter/workstreams.json`. This record describes the requested repairs, not completed
second-pilot results.

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
