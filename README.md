# Cross-Lingual Duplicate Question Detection (en / hi / mr)
See CLAUDE.md for the full spec. Status: Phases 0-3 done; Phase 4+ pending.

## Tooling constraints
- **Translation model: `facebook/nllb-200-distilled-600M` (NLLB), not IndicTrans2.** IndicTrans2 is gated and no
  HF token/terms acceptance was available. NLLB-200 is **CC-BY-NC-4.0**: the generated Hindi/Marathi data inherits a
  non-commercial restriction.
- Translation was run on a Colab T4 (`colab.ipynb`, fp16, 5 beams); local CPU (5 GB RAM) was infeasible.
- Local env: `.venv`, Python 3.12, CPU torch; torch 2.14.1, transformers 4.57.6, datasets 5.0.1, sentence-transformers 5.7.0.

## Run
`make smoke` (50-pair pipeline). Full: `cd src && python s00_sample.py && python s02_translate.py && python s01_lexical.py && python s04_evaluate.py`.

## Demo
`.venv/bin/python app.py` then open http://127.0.0.1:7860 (IndicSBERT, threshold = en-en tau*; examples are real test pairs, selected from correctly classified ones).
