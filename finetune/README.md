# Phase 8: fine-tuning (isolated copy)

Everything here is separate from the frozen zero-shot baseline in the repo root. `config.py` resolves paths relative
to this folder, so nothing in here can modify the baseline.

Plan (CLAUDE.md Phase 8): fine-tune `l3cube-pune/indic-sentence-similarity-sbert` with `OnlineContrastiveLoss`;
24,000 balanced QQP-`train` pairs (leak-checked against dev/test), each shown in one randomly chosen language
condition from {en-en, hi-hi, mr-mr, en-hi}; 2 epochs, batch 32, max_seq_length 64, lr 2e-5, 10% warmup, fp16.

## Runs (2 Colab runs, T4 GPU)
| # | Notebook | Bundle in -> out | Time |
|---|---|---|---|
| local | `s10_prep_train.py` | (done) writes `data/processed/train_pairs.parquet` (24k pairs, 44,439 unique questions) | seconds |
| 1 | `colab_ft1_translate.ipynb` | `ft1_bundle.zip` -> `ft1_out.zip` (translated train pairs + cache) | ~75 min |
| 2 | `colab_ft2_train.ipynb` | `ft2_bundle.zip` (built after run 1) -> `ft2_out.zip` (embeddings + `ft_run.json`) | ~15-25 min |
| local | `s13_eval_ft.py` | scores `indicsbert-ft`, same protocol/splits; writes `results/ft_vs_base.csv` | seconds |

After run 1: unzip `ft1_out.zip` here (replaces `data/processed/train_pairs.parquet` and the cache), then ask for `ft2_bundle.zip`.
After run 2: unzip `ft2_out.zip` here, then `cd src && ../.venv/bin/python s13_eval_ft.py`.

Smoke-tested locally on tiny data (all four scripts); the training step was verified to change the weights.
