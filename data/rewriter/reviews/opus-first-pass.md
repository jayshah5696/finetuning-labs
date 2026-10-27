# Opus data verification, first pass

Verifier: Opus 5.5 through the Pi provider. Scope: data only, no generation or code review.
Task: `node:delegated-task:command%3Amcp%3Ab224eb97-2a74-45f4-a035-38e5ac7cebc4%3Adelegate-task%3A92223b25-1490-4805-b4a0-e79a1dda3510`.

Verdict: the 219-pair synthetic pilot is not high quality and should not be approved for training.
The verifier inspected 12 training/internal-validation pairs and four short benchmark references.
This is a sampled review, not a review of every row.

## Concrete findings

- `bicycle-transcribe-challenge`: pass, but a weak challenge that removes one repeated phrase.
- `microphones-prompt-mode-noise`: uncertain; permission for an additional action becomes an
  instruction to ask for it. Authoring wording about quoted data leaked into the text.
- `astronomy-slack-challenge`: pass; minimal grammatical cleanup.
- `geology-polish-light`: pass but trivial; only a comma changes.
- `sailing-transcribe-long-v3-challenge`: uncertain; sparse added fillers do not resemble real ASR.
- `board-games-slack-long-v3-light`: fail as training signal; only double spaces differ.
- `ceramics-polish-long-v3-challenge`: fail on input realism; repeated article deletion and padded
  target prose. An unspecified additional action is discussed repeatedly.
- `coin-club-prompt-long-v3-mode-noise`: pass but mechanical; only paragraph/list breaks change.
- `bird-count-prompt-identity`: pass as identity, with authoring rules leaked into ordinary content.
- `choir-slack-long-v3-identity`: uncertain; one very long repetitive Slack message is implausible.
- `dog-daycare-transcribe-challenge`: uncertain; ambiguous Portuguese self-correction conflicts
  with a literal rule that forces the earlier quantity to survive.
- `bookbinding-transcribe-mode-noise`: pass with a defect; German wording changes exceed cleanup.

The four sampled short benchmark references passed. Meaningful German words such as `er` and
`um` were correctly preserved by those authored benchmark targets.

## Distribution defects

Opus found that all 55 source targets follow the same count-checking scenario. Four variants
share each target. Inputs rarely start lowercase, unlike the benchmark's dictation-like text.
Long documents stretch a few facts rather than supply useful new information. Noise is often
spacing, global word deletion or isolated fillers. The dataset lacks enough realistic punctuation,
speech repairs, dictionary fixes, and meaningful changes appropriate to each mode.

These findings supersede any earlier implication that automatic passes certified this pilot.
The replacement must be independently verified and may not inherit these automatic approvals.
