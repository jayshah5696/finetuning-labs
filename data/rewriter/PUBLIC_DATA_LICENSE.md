# Data sources and rights

The dataset is public for browsing and download. It contains shared fictional and authored
examples. It contains no personal TypeWhisper history, recordings or private transcription exports.

Training and development examples are synthetic pairs authored for this project. Existing model
reviews are quality evidence, not human labels or a separate license grant. The dataset is marked
`other` because its source material has mixed rights. Do not assume an Apache-2.0 grant for the
whole dataset.

Thirty original validation cases come from authored TypeWhisper Linux fixtures. That source is
GPL v3. The original source snapshot, source revision and attribution remain in the Git project
under `data/rewriter/benchmark-draft/supporting/`. The complete source license is included here
at `licenses/TYPEWHISPER-LICENSE.txt`.

The other original benchmark cases and the later synthetic pairs were authored for the rewriter
project. The current training dataset's 84 validation rows preserve the reviewed-v2 references.
The separate historical calibration dataset preserves the original 68-case references.

Code, notebooks, source snapshots and verification records are maintained in
https://github.com/jayshah5696/jadoo-labs/tree/main/projects/finetuning-labs .
