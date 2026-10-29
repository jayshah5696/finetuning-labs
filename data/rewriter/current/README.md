# Current rewriter training data

1,500 train, 100 dev and 84 frozen validation pairs. The original 1,000 training rows remain unchanged; 500 independently generated Arka episodes target observed fidelity failure types. Dev and validation bytes are unchanged.

New pairs use Luna generation and six Luna Decisions checks. Complete uncertain native scores require two separate exact-row GPT-6.1 Sol passes. Strong native decisions require one. Confident native rejection cannot be overridden. Each verification checks the input, output and pair. These are model judgments; human calibration remains pending.

Hugging Face contains only public Parquet splits, a dataset card and licenses. Code, notebook and exact review evidence stay in Git. Use the single notebook at `../../../notebooks/rewriter_train_l4.ipynb`.
