# Stock Liquid diagnostic baseline

This report measures declared mechanical checks. It is not a semantic quality score.
Human calibration and the Sol judge run remain pending.

Model: `LiquidAI/LFM2.5-230M` at `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
Runtime: NVIDIA L4 on Modal, PyTorch 2.13.0, BF16, eager attention, greedy decoding.
The reference convolution path was used because causal_conv1d was not installed.
This is a functional baseline, not an optimized latency benchmark.

| Mode | Scheduled | Mechanical passes |
| --- | ---: | ---: |
| /slack | 12 | 8 |
| /transcribe | 32 | 26 |
| /prompt | 12 | 10 |
| /polish | 12 | 9 |

Total: 53/68 mechanical passes.
Generation outcomes: {'completed': 68}.
Long cases: 0/8 mechanical passes.
Explicit identity cases: 0/3 unchanged.

Every successful mechanical result still needs semantic review. A normal end token
does not prove that every paragraph survived. The long outputs include missing quoted
content, shortened documents, and added commentary. Preserve the raw outputs for review.

Luna and Sol are not reported as evaluated without actual provider calls.
