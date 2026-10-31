# English rewriter data workstreams

The user approved separate threads for the next experiments. Data work starts now. Training
rewards, LoRA ablations and preference training remain deferred. The student models remain
LiquidAI/LFM2.5-230M and LiquidAI/LFM2.5-350M. Qwen is an optional later comparison, not a selected
model or a current download/training task.

This is the shared handoff contract. Thread links and branches are recorded in
`data/rewriter/workstreams.json` after creation. Each active thread owns its worktree and its
staging namespace. No worker merges to main or changes the active Hub release independently.

## Common rules

- English only. Keep the existing four rewrite modes and their exact contracts.
- Preserve `data/rewriter/current`, all earlier train rows, the original source holdouts, the
  81-case English validation view and the existing private calibration set. Stage additions.
- Reuse native Arka generation, all six Luna Decisions dimensions and the existing exact-row
  Sol review loop. The verified Arka HEAD and current pin are both
  `5d353740ae1dd34742841071752814028db1489a`. Do not change the shared pin independently.
- Use the existing authorized OpenRouter credential without printing or saving its value.
  Luna authors synthetic candidates. GPT-6.1 Sol verifies inputs, outputs and pairs. Preserve
  native decisions, raw calls, rejected attempts and exact hashes. No new Opus calls.
- Keep failed and uncertain reviews excluded. A strong native admission needs a Sol pass;
  other complete, non-vetoed admissions need two Sol passes under the existing policy.
  These are model reviews, not human gold labels. Do not reroll completed verdicts.
- Keep viewer rows to input, output, mode and checks. Keep SFT rows to system, user and
  assistant. Provenance, rule labels, review scores and privacy mappings stay outside the
  supervised text. Do not change this format or introduce a new dataset repository/version.
- Long inputs should contain 1,000 to 2,000 pinned Liquid tokens with substantive context,
  paragraphs, numbered material and quotations where natural. Check both student tokenizers,
  full chat fit and the 2,048-token generation reserve. Reject overlength rather than truncate.
- Distinguish a **transformation rule** from a **source family**. The same rule belongs in
  training and validation. Variants of the same source document or scenario belong in one split.
- Never copy validation text into training or make name-swapped versions of a validation
  document. Allocate independent episode/source families to splits before generation. Training
  generation prompts must not contain the validation documents or reference answers.
- Run a bounded pilot and inspect coverage, repetition and noise before scaling. Stop expansion
  if the quality gates fail; correct the generator or use fresh independent sources. Report
  shortfalls instead of claiming the intended counts are complete.
- Start from the existing native pipeline and admission/export helpers. Add small task-specific
  configs or wrappers. Do not build another generation/review framework.
- Write focused failure cases/tests before production code. Run relevant tests, `make check`
  and `make build`, and exercise the actual staged export. Keep evidence and scripts in Git.
- No student training, notebook changes, rank sweeps, new model downloads or preference training
  in the active data threads. Tokenizer downloads and small test fixtures are permitted.

## Initial transformation-rule identifiers

These identifiers provide a common vocabulary. Thread 1 may refine descriptions and criteria,
but should preserve the identifiers so the other threads can map their coverage.

| ID | Rule | Important controls |
| --- | --- | --- |
| R01 | Explicit replacement and chained corrections | Follow final intent; retain unrelated numbers, names, time and participants. |
| R02 | Canceled actions and replacement requests | Remove canceled actions without removing the surviving request. |
| R03 | Negation, prohibitions and exclusions | Preserve the forbidden action and the separately requested positive output. |
| R04 | Conditions, scope and uncertainty | Keep actors, branches, prerequisites, hedges and source boundaries. |
| R05 | Deliberately unfinished thoughts | Do not invent an action, outcome or conclusion. |
| R06 | Disfluencies versus meaningful words | Remove true restarts; preserve acknowledgments and meaningful discourse words. |
| R07 | Literals, dictionary terms and spoken identifiers | Preserve or explicitly normalize names, code, URLs, quantities and filenames. |
| R08 | Task intent and rewrite mode | Organize the request without executing it or inventing requirements. |
| R09 | Long-document completeness and structure | Preserve every constraint, final exception, quotation and relevant section. |
| R10 | Minimal edits and unchanged inputs | Include plausible short and long cases where the correct target is identical. |

## Thread 1: transformation rules and Arka training pairs

**Start now.** Own the rule definitions, contrasts and independent synthetic training examples.

- Define each rule's applicable modes, allowed edits, forbidden edits, ambiguity exclusions and
  positive/negative controls. Teach final-intent changes rather than requiring every old literal
  to survive an explicit correction.
- Generate short and long training pairs for every applicable rule. Start with a pilot of two
  short and one long pair per rule. Once its coverage and exact reviews pass, target ten short
  and five long accepted pairs per rule. Preserve actual counts when slots remain unfilled.
- Use fresh episode/source families and balanced correction directions. Include near-neighbor
  controls where removing the earlier word or number would be wrong. Avoid a repeated template.
- Own `configs/data-expansion/transformation-rules.*`, `docs/transformation-rules.md`, minimal
  task-specific code/tests named for rule expansion, and
  `data/rewriter/expansions/transformation-rules/`. Use `rules-*` source IDs.
- Deliver a reviewed staged data/SFT bundle, rule/mode/length coverage, exclusions, reproduction
  commands and a handoff report. Do not append to the active release or publish it independently.

## Thread 2: personal English dictation, shortcuts and redaction

**Start now, locally.** The user authorizes using their personal English material and removing
obvious PII. The existing raw-personal-data restrictions continue to protect the source records.

- Inspect the existing ignored personal bundle at the main checkout's
  `projects/finetuning-labs/data/rewriter/personal/calibration-10/`, its source manifest, and the
  read-only TypeWhisper repository at `/home/jshah/Documents/GitHub/typewhisper-linux`.
  Inspect only related app history/config paths discovered from those sources.
- Keep the existing ten calibration records held out. Select additional records covering actual
  dictation errors, rewrite shortcuts/profiles, shorthand and supplied dictionary behavior.
  Do not confuse hotkey configuration with text transformations or infer an unknown mode from
  output style. Historical app output is evidence, not an approved correction.
- Work on copies. Never edit the app's history database, settings, shortcuts or dictionaries.
  Remove credentials, names, contact details, user paths and identifying context. Use distinct,
  consistent fictional replacements so intended and rejected entities do not collapse into one
  placeholder. Preserve the transformation, numeric relationships, negation and conditions.
- Keep raw records, direct redacted records, before/after reviews and identity mappings local
  under ignored `data/rewriter/personal/` or `artifacts/`. Do not send raw personal records to
  hosted models, Git, Hugging Face or Sangam. Obvious-PII removal is not proof of anonymity.
- Produce a local redacted candidate set and a non-identifying error-pattern inventory. Public
  staged candidates may be independently fictionalized analogues of those patterns. Direct
  personal-derived rows remain local for review; no public release of them is part of this thread.
- Own `docs/personal-english-data.md`, minimal local redaction/import code/tests, ignored private
  artifacts and any fully fictionalized staged rows under
  `data/rewriter/expansions/personal-patterns/`. Use `personal-patterns-*` synthetic source IDs.
- Report coverage, redaction checks, reference provenance and unresolved ambiguity without raw
  examples or private source IDs in the shared handoff. Do not label model reviews as human approval.

## Thread 3: long-example validation and matching training coverage

**Start now.** Own the new long-example evaluation design and its independent training analogues.

- Map every one of the existing 81 English validation cases to its transformation/constraint
  checks and a plausible long-form expansion. Keep the original cases byte-identical.
- Define the long-example checks before authoring targets: intended and canceled actions,
  facts and actors, negation, conditions, uncertainty, literals, quotes, structure and closing
  exceptions. Review every required constraint, not just an overall similarity score.
- Plan one fresh long validation episode per existing case, with independent new scenarios.
  Give each new validation criterion at least three independent training analogues spanning
  short and long forms. Shared rules are expected; shared source documents are forbidden.
- Allocate and freeze the new validation source groups before training generation. Training
  generation must receive only rule/criterion briefs and its own independent episode facts.
  Keep new validation separate from the current frozen 81-case bundle and active Hub release.
- First verify a pilot covering each rule family, then expand toward the full 81-case plan if
  the pilot passes. Make long Slack examples plausible threads, keep prompt outputs as requests,
  and spread genuine editing challenges through long inputs. Do not pad a short fact into an essay.
- Own `docs/long-example-coverage.md`, `configs/data-expansion/long-coverage.*`, minimal
  long-coverage code/tests, `data/rewriter/expansions/long-coverage/` and a staged long-validation
  bundle within that directory. Use `long-coverage-*` source IDs with explicit split roles.
- Deliver the 81-row coverage map, split allocation, constraint checklists, accepted data/SFT
  rows, exact reviews and exclusions. Label fresh references as model-reviewed until human review.

## Thread 4: training rewards and checkpoint selection

**Parked.** Capture the future questions only. No sampling, loss, checkpoint policy or notebook
changes now. After data readiness, compare training-token balance, dev rewrite quality and loss
curves. Checkpoints must be selected with dev, never with the validation comparison.

## Thread 5: LoRA rank and model ablations

**Parked until data readiness.** Plan ranks 16, 32 and 64 for both 230M and 350M using the same
reviewed English data. Separate rank, scaling, seed, token budget and checkpoint-selection
effects. Preserve model/revision/adapter-destination coupling in the one existing notebook.
Qwen is optional; its exact candidate, license, chat template and token limits need research
when this workstream is activated. No Qwen selection or training is authorized by this parked brief.

## Thread 6: preference training after supervised fine-tuning

**Parked until supervised fine-tuning is assessed.** Define what would justify preference
training after the SFT data and ablation work. Potential preference pairs use independently
reviewed targets and actual student failures on new training prompts. Never use validation
answers as preference-training examples. No DPO/GRPO run or reward-model build now.

## Integration owned by the coordinating thread

Merge reviewed workstreams sequentially. Run cross-stream duplicate/source-family checks,
private-holdout overlap checks, token audits, reference/review binding checks and privacy checks
from the main checkout before release. Verify old training prefixes and all frozen holdouts.
Apply the exact-row exclusions in `data/rewriter/expansions/coordinator/quarantine.json` before
selecting staged rows. Preserve the original rows and passing model verdicts as evidence.
Update the same public dataset only after the combined release passes. Keep direct personal
records private. Update the shared notebook's expected counts only when the released data is ready.
Then activate the deferred training work. Do not claim that data is complete merely because
the threads were created or a target count was proposed.
