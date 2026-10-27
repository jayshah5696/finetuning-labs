# Rejected for training after Opus data verification

Do not use this pilot as a high-quality training release. Opus's first data-only verification
found a repeated count-checking template, unrealistic synthetic errors, padded long documents,
authoring rules leaked into ordinary text, and several changes of nuance. The original machine
scores remain historical evidence; they did not establish data quality.

The published bytes and old run receipts are preserved for reproducibility. The replacement
uses the latest Arka pipeline, independent source briefs, native input/output/pair criteria,
and a separate Opus verification loop. See `data/rewriter/reviews/opus-first-pass.md`.
