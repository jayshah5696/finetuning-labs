# Current rewriter training sources

2,000 source training pairs, 100 dev and 84 frozen validation pairs. This append adds 500 independently generated Arka episodes. All earlier training rows remain unchanged; dev and validation source bytes are unchanged. The active public export selects English only.

New pairs use Luna generation and six Luna Decisions checks. Complete uncertain native scores require two separate exact-row GPT-6.1 Sol passes. Strong native decisions require one. Confident native rejection cannot be overridden. Each verification checks the input, output and pair. These are model judgments; human calibration remains pending.

Hugging Face contains only public Parquet splits, a dataset card and licenses. Code, notebook and exact review evidence stay in Git. Use the single notebook at `../../../notebooks/rewriter_train_l4.ipynb`.
