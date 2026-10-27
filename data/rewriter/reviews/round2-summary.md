# Opus second-pass data verification

The data-only reviewers inspected all 161 submitted pairs. Training and development had 79
pairs: 63 pass, six fail, and ten uncertain. Validation had 82 pairs: 75 pass, six fail, and one
uncertain. The validation group includes all 68 original references and 14 new candidates.

The original references had one punctuation-consistency defect and one ambiguous source. The
new candidate defects included dropped quotations or conditions, literal-key punctuation,
transcription paraphrases, leaked formatter headers, and a source with contradictory counts.

The parent prepared 23 corrected rows and retained the unchanged rows with their existing exact
hashes. Three complete length-repair examples and four new medium/long identity examples were
added. All 168 resulting pairs were checked by the native Arka pipeline and fit the Liquid token
limits. Thirty exact rows require a fresh data verification: 20 train/dev and ten validation.

These are model judgments, not human gold labels. The parent checked a mistaken corpus-level
claim in the validation report: the eight original long cases come from four source families
and four distinct targets, not two. The narrower point that related long variants share source
documents remains valid. The old 68-case files and their prior model scores are unchanged.
