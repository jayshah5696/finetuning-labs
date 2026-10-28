# Sol review context

The passing Sol report is `../../reviewed-v3-1000/runtime/script-review.json`.
Its file hashes describe the exact inputs at review time. This folder preserves the small
pre-publication records and the agent instructions that later received a release-status update.
`paths.json` maps those original paths to their retained locations. Other source and test paths
stay unchanged. The final supporting.json is byte-identical to the reviewed preparation.

All six final data and SFT files were compared byte for byte with the reviewed preparation.
The final private Hub files also passed the actual notebook download and runtime readiness cells.
The full temporary preparation was archived locally under ignored artifacts. It is not a second
training dataset and does not need to be duplicated in the repository.
