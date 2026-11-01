# Current rewriter training sources

2,115 training rows, 100 dev rows, and 144 validation rows (2,359 total rows). This release includes the 115 diverse-v2 training additions and 60 validation-v1 validation additions across the 5 dictation families and 500–2,000 token range: developer coding symbols, workspace file context (@file tags), spoken list structures, multi-level contrasts, and chained speech repairs / anti-chatbot isolation. All earlier training rows remain unchanged. The active public export selects English only.

New pairs use Luna generation and six Luna Decisions checks. Complete uncertain native scores require two separate exact-row GPT-6.1 Sol passes. Strong native decisions require one. Confident native rejection cannot be overridden. Each verification checks the input, output, and pair. These are model judgments; human calibration remains pending.

Hugging Face contains only public Parquet splits, a dataset card and licenses. Code, notebook and exact review evidence stay in Git. Use the single notebook at `../../../notebooks/rewriter_train_l4.ipynb`.
