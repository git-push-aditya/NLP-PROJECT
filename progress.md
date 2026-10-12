# Progress log

Living file: updated after every phase. Newest status first, details per phase below.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Scaffold + smoke harness | Done |
| 1 | Data acquisition + splits | Done |
| 2 | Lexical (TF-IDF) baseline | Done |
| 3 | Translation (en -> hi, mr) | Done (resume test skipped by choice) |
| 4 | Embedding + evaluation harness | Done |
| 5 | Diagnostics / error analysis | Done |
| 6 | Out-of-domain check (MahaParaphrase) | Done |
| 7 | Gradio demo | Done |
| 8 | Fine-tuning (in `finetune/`) | In progress: scripts and Colab notebooks ready, run 1 of 2 not yet done |

Deferred by decision: kill-and-resume test of translation; git commit of the work (nothing committed yet, only the initial commit exists).

---

## Phase 0: Scaffold and smoke harness

**Built:** `src/config.py` (all constants), `Makefile` (`make smoke`), `--smoke` flag on every script, `.venv` (Python 3.12, CPU torch 2.14.1, transformers 4.57.6, datasets 5.0.1, sentence-transformers 5.7.0), `src/s_minus1_verify.py` (acquisition asserts, passes).

**Output:** `make smoke` runs end to end on 50 test + 30 dev pairs and writes `results/metrics_smoke.csv` with the right columns. Smoke artifacts use a `_smoke` suffix and never overwrite full-run files. The smoke set is a subset of the full pairs, so it reuses the full translation cache and loads no model.

## Phase 1: Data and splits

**Output:** dev 1,000 and test 2,000 pairs from QQP `validation`, each balanced at label mean 0.5, no pair overlap (checked), seed 42. `data/interim/pair_ids.json` and `data/processed/pairs_all.parquet` written. Bucket counts:

| split | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| dev | 230 | 334 | 203 | 148 | 85 |
| test | 458 | 675 | 442 | 285 | 140 |

**Caveat:** the "same pair IDs on re-run" check was only verified as "script skips if file exists", not by regenerating from scratch.

## Phase 2: Lexical baseline

**Output:** char_wb 3-5-gram TF-IDF cosine, tau swept on dev, applied to test.

| Condition | F1 | AUC | tau* | degenerate |
|---|---|---|---|---|
| en-en | 0.753 | 0.765 | 0.36 | no |
| hi-hi | 0.711 | 0.718 | 0.27 | no |
| mr-mr | 0.718 | 0.717 | 0.24 | no |
| en-hi | 0.667 | 0.542 | 0.00 | yes |
| en-mr | 0.667 | 0.517 | 0.00 | yes |
| hi-mr | 0.667 | 0.605 | 0.00 | yes |

`all-positive` floor row present (F1 0.667, AUC 0.50). TF-IDF `en-en` F1 (0.753) is above the 0.60-0.70 band the spec expected, so it is a strong baseline.

## Phase 3: Translation

**Output:** all 12,000 sentences (5,906 unique per language, 11,812 cache rows) translated with NLLB-200 600M on a Colab T4. No nulls or empties; about 99% of each column is Devanagari (the rest are names and acronyms in Latin script). Cache in `data/processed/translation_cache.sqlite`; a re-run is a no-op (about 9 s, no model load).

**Adequacy audit** (`results/adequacy_audit.csv`, 60 pairs, 20 per language-pair group, scorer Aditya-D, 1-3 scale):

| | Mean | Scored 1 |
|---|---|---|
| Hindi | 2.78 | 2 of 60 |
| Marathi | 2.60 | 7 of 60 |

Serious MT errors found: "weed" translated as the plant (hi `खरपतवार`, mr `गवत`); "falcon" became tiger (hi) and a wrong word (mr); "Probationary Officer" became "suspension officer" (mr); "redeemed" became "readymade" (mr); "diet and exercise" lost "exercise" (mr); a dropped "physics" clause (hi and mr); dropped content in mr ("drive from Edmonton to Auckland", "bounty" as "prizes"). Marathi is noticeably weaker than Hindi. Quote these means next to every Indic number.

**Tooling constraint:** IndicTrans2 is gated and no HF token was available, so the NLLB fallback was used. NLLB is CC-BY-NC-4.0, so the generated Hindi/Marathi data inherits a non-commercial restriction (recorded in `README.md`, `results/run_config.json`).

## Phase 4: Embeddings and evaluation

**Built:** `s03_embed.py` (encode each language column once per model, cache normalised `.npy`, assert norms), `s04_evaluate.py` (threshold protocol, metrics, 1,000x bootstrap CIs, degenerate flag, writes per-pair scores to `data/processed/scores.parquet`). Embeddings computed on Colab T4 (`colab_phase4.ipynb`), 18 files in `data/processed/emb/`; scoring runs locally in seconds.

**Acceptance:** met. IndicSBERT `en-en` AUC 0.845 beats TF-IDF 0.765. `metrics.csv` has 54 rows each for TF-IDF, IndicSBERT, LaBSE, mpnet (6 conditions x 9 metrics) plus 9 for the floor.

**Test F1 / AUC:**

| Condition | IndicSBERT | LaBSE | mpnet | TF-IDF |
|---|---|---|---|---|
| en-en | 0.786 / 0.845 | 0.768 / 0.808 | **0.817 / 0.880** | 0.753 / 0.765 |
| hi-hi | 0.779 / 0.838 | 0.765 / 0.805 | 0.781 / 0.846 | 0.711 / 0.718 |
| mr-mr | 0.763 / 0.825 | 0.758 / 0.795 | 0.766 / 0.821 | 0.718 / 0.717 |
| en-hi | 0.767 / 0.833 | 0.759 / 0.792 | 0.782 / 0.841 | 0.667 / 0.542 (degenerate) |
| en-mr | 0.764 / 0.816 | 0.757 / 0.794 | 0.760 / 0.810 | 0.667 / 0.517 (degenerate) |
| hi-mr | 0.770 / 0.818 | 0.757 / 0.800 | 0.769 / 0.825 | 0.667 / 0.605 (degenerate) |

`en-en` 95% F1 CIs: IndicSBERT 0.765-0.804, LaBSE 0.750-0.786, mpnet 0.800-0.834, TF-IDF 0.735-0.771.

**Findings:**
- Encoders beat TF-IDF everywhere; cross-script cells carry real signal (AUC 0.79-0.84, none degenerate).
- Leaving English costs only about 2-4 AUC points, less than the spec predicted. IndicSBERT degrades least (0.845 to 0.816 on `en-mr`). Consistent with the MT optimistic-upper-bound caveat: the Indic side is a translation of the English.
- mpnet leads `en-en`, but gaps under 4 F1 points are not distinguishable (CIs about +/-2). mpnet vs IndicSBERT (0.031) is not a clear win.
- `tau*` drifts down for cross-lingual pairs (IndicSBERT 0.67 on en-en to 0.55 on en-mr): ranking holds, a single global threshold does not.
- The primary model (IndicSBERT) did not win on English; mpnet did. State this plainly in the report.
- `en-en` F1 0.786 is slightly above the 0.68-0.78 band; AUC 0.845 is at the top of its band. Nothing looks wrong.

## Phase 5: Diagnostics and error analysis

**Built:** `src/s05_diagnose.py` (reads cached scores plus the frozen dev `tau*`; no GPU). Outputs: `results/figures/bucket_f1.png`, `results/figures/tau_drift.png`, `results/slices.csv`, `results/error_examples.md` (10 FP + 10 FN, IndicSBERT `en-en`, most confident errors, hand-written diagnoses in `results/error_diagnoses.json`). TF-IDF cells that are degenerate are drawn dashed/hollow and left out of the tau plot.

**Deviation from spec:** the "B5 negatives precision" slice is reported as the **false-positive rate** (precision is undefined when the slice has only negatives).

**Key slices (test; B5 negatives n=54, B1 positives n=74):**

| Model / condition | B5 neg FPR (lower is better) | B1 pos recall |
|---|---|---|
| IndicSBERT en-en | 0.76 | 0.73 |
| IndicSBERT en-mr | 0.85 | 0.81 |
| LaBSE en-en | 1.00 | 0.76 |
| mpnet en-en | 0.87 | 0.81 |
| mpnet en-mr | 0.72 | 0.82 |

**Findings:**
- **Lexical shortcutting:** all three encoders call 72-100% of high-overlap non-duplicates (B5 negatives) duplicates. A good overall F1 is partly overlap, not meaning. LaBSE is the worst (100%).
- **B1 positives:** low-overlap true duplicates are caught about 70-82% of the time, so there is real semantic matching, but F1 on B1 is far below B3/B4 (about 0.5-0.7 vs about 0.8).
- **Other slices (en-en, IndicSBERT):** negation mismatch FPR 0.19 (best of the three models); number mismatch recall 0.60 (mpnet 0.86, so IndicSBERT is the weakest on numbers); under-5-token questions FPR 0.48 with recall 1.00 (short questions are over-called duplicates).
- **Threshold drift:** `tau*` falls for cross-lingual conditions for every encoder; the drift pattern differs by model (LaBSE is non-monotonic).
- **Error examples:** about half the top false positives are near-identical rewordings ("the video" vs "this video"), likely QQP label noise rather than model failure. The clear model failures are numeric substitution (3 vs 1.3 years) and modality/tense shifts. The false negatives are mostly genuine paraphrase with low overlap, or loosely-labelled pairs.
- Caveat: slice sizes are small (n=54 to 227), so differences between models in a slice are noisy; treat as indicative.

## Phase 6: Out-of-domain check (human-written Marathi)

**Built:** `src/s06_ood.py`. Uses MahaParaphrase's own splits (tau tuned on `validation` n=999, reported on `test` n=2,000; `train` ignored, no training). Labels mapped `{"P":1,"NP":0}`. Embeddings computed on Colab (`colab_phase6.ipynb`), 6 files `data/processed/emb/ood_*.npy`; scoring is local. Rows are named `ood-<model>` with split `ood_test`, so they are excluded from headline aggregates and `s04_evaluate.py` never overwrites them (checked: 0 headline rows carry the OOD split).

**Test set is balanced (1,000 P / 1,000 NP)**, so the all-positive floor is F1 0.667 / AUC 0.50 as before.

| Model | F1 | AUC | Accuracy | tau* | F1 95% CI low |
|---|---|---|---|---|---|
| IndicSBERT | 0.790 | 0.830 | 0.744 | 0.75 | 0.771 |
| LaBSE | 0.780 | 0.834 | 0.728 | 0.70 | 0.762 |
| mpnet | 0.780 | 0.811 | 0.738 | 0.80 | 0.762 |
| TF-IDF | 0.730 | 0.654 | 0.642 | 0.17 | 0.711 |
| all-positive | 0.667 | 0.500 | 0.500 | n/a | 0.646 |

**Findings:**
- On human-written (non-translated) Marathi the encoders reach AUC 0.81-0.83, in line with our machine-translated `mr-mr` result (AUC 0.795-0.825). So the Indic results are **not just a machine-translation artefact**; translationese is not what makes the encoders look good.
- The encoders are not distinguishable from each other here (F1 gaps about 0.01, CIs about +/-0.02).
- TF-IDF is far weaker on AUC (0.654), consistent with Phase 4.
- Zero-shot F1 0.78-0.79 is below the anchor paper's fine-tuned cross-encoders (84.6-88.7 F1), as expected for frozen embeddings with a single threshold; the setups differ, so this is context, not a head-to-head.
- Framing, state explicitly in the report: MahaParaphrase controls for **translationese** but not for **question-ness** (it is news-domain declarative sentences). It is a control, not training data, and is kept out of the headline table.

## Phase 7: Demo

**Built:** `app.py` (Gradio 6.17.3). Loads IndicSBERT once at module scope, shows the raw cosine score, the Duplicate / Not duplicate label, and the threshold used (the `en-en` `tau*` = 0.67, read from `metrics.csv`, not hardcoded). Run: `.venv/bin/python app.py`, then open `http://127.0.0.1:7860`.

**Seeded examples** (real held-out QQP test pairs, one per condition; chosen at startup from cached scores: correctly classified, texts differ, moderate margin so they are not trivially easy; labels alternate):

| Condition | Truth | Score | Prediction |
|---|---|---|---|
| en-en | duplicate | 0.820 | Duplicate |
| hi-hi | not duplicate | 0.519 | Not duplicate |
| mr-mr | duplicate | 0.820 | Duplicate |
| en-hi | not duplicate | 0.519 | Not duplicate |
| en-mr | duplicate | 0.819 | Duplicate |
| hi-mr | not duplicate | 0.520 | Not duplicate |

All six classified correctly. **Be upfront in the viva:** these examples were selected from pairs the model gets right (overall accuracy on each condition at the en-en threshold is about 0.74-0.76, so roughly one in four pairs is wrong). hi/mr text is machine-translated.

**Checked:** the app launched and answered HTTP 200 locally. Model load about 10 s; about 0.1-0.7 s per prediction on CPU (first call slowest); about 1 GB RAM while running.

---

## Compute and timing (for the demo / viva)

**Local machine:** AMD Ryzen 5 5500U, 12 threads, 5.7 GB RAM, no GPU, Linux (Fedora), Python 3.12 venv. Free Colab T4 GPU used for the heavy encoding/translation.

**Software:** torch 2.14.1, transformers 4.57.6, sentence-transformers 5.7.0, datasets 5.0.1, gradio 6.17.3.

| Step | Where | Time |
|---|---|---|
| Phase 3 translation (11,812 unique sentences, NLLB-600M, fp16, 5 beams) | Colab T4 | about 10 min (Hindi about 5 min, Marathi about the same) |
| Same translation attempt | Local CPU | abandoned after 83 CPU-min (stalled, swap full) |
| Phase 4 embeddings (3 models x 6 columns x 3,000) | Colab T4 | a few minutes (not timed) |
| Phase 6 OOD embeddings (3 models x about 6,000 sentences each) | Colab T4 | a few minutes (not timed) |
| `s01_lexical.py` (TF-IDF, all conditions) | Local CPU | 9.6 s, 0.32 GB peak |
| `s04_evaluate.py` (6 conditions x 3 models + bootstrap CIs) | Local CPU | 2.9 s, 0.38 GB peak |
| `s05_diagnose.py` (slices, 2 plots, error examples) | Local CPU | 10.0 s, 0.40 GB peak |
| `s06_ood.py` (scoring from cached embeddings) | Local CPU | 15.7 s, 0.98 GB peak |
| Demo: model load / per prediction / RAM | Local CPU | about 10 s / about 0.1-0.7 s / about 1 GB |

**Data size:** 3,000 pairs (1,000 dev, 2,000 test) x 2 questions x 3 languages. Encoders: IndicSBERT (about 238M params), LaBSE (about 471M), paraphrase-multilingual-mpnet (about 278M); translation model NLLB-200 distilled 600M. No training anywhere (zero-shot, frozen).

## Phase 8: Fine-tuning (in progress, isolated in `finetune/`)

**Safety:** the finished Phases 0-7 were committed and pushed (private repo `git-push-aditya/NLP-PROJECT`, commit `fb07b83`) before any fine-tuning work. All fine-tuning lives in `finetune/`, a copy whose `config.py` resolves paths to that folder, so the baseline cannot be modified by it.

**Built and smoke-tested locally (tiny data):** `s10_prep_train.py` (24,000 balanced pairs from QQP `train`; 16,592 train pairs dropped for sharing a question with dev/test, so zero leakage; 44,439 unique questions to translate), `s11_translate_train.py` (reuses NLLB + sqlite cache), `s12_finetune.py` (OnlineContrastiveLoss, 2 epochs, batch 32, max_seq 64, lr 2e-5, 10% warmup; then encodes dev/test and MahaParaphrase with the tuned model), `s13_eval_ft.py` (same protocol as the baseline; F1/AUC plus B5-negative FPR and B1-positive recall vs frozen IndicSBERT; writes `ft_vs_base.csv`).

**Needs 2 Colab runs (T4):** run 1 translate (about 75 min), run 2 train + encode (about 15-25 min). See `finetune/README.md`.

**Bug caught by the smoke test:** the HF trainer needs `accelerate`, not in our requirements; installed locally and added to the Colab notebook's pip line. Also verified the identical fine-tuned/baseline smoke scores were due to a zero learning rate on a single warmup step, not a pipeline bug: with a forced real update the embeddings changed (cosine to baseline 0.49).

**Expectation to check after training:** the zero-shot result and its clean baseline are the reference point; watch whether fine-tuning raises AUC but also worsens B5-negative false positives (more lexical shortcutting).

---

## Problems faced and how they were handled

| # | Problem | Cause | Resolution |
|---|---|---|---|
| 1 | Python deps missing; background install silently did nothing | Nothing installed on the machine; the backgrounded `uv pip install` produced an empty log | Re-ran the install in the foreground; created `.venv` (Python 3.12, since system Python is 3.14) |
| 2 | Local NLLB translation stalled | 600M model in fp32 with 5 beams on a 5 GB RAM, no-GPU machine; swap filled (about 3.9 GB), cache frozen at 128 Hindi sentences after 83 min CPU time | Killed the run; moved translation to a Colab T4 (added auto GPU and fp16 to `s02_translate.py`). Full run took about 10 min |
| 3 | Monitors/waits expiring with no output | Waiting on a stalled job | Checked the sqlite cache count directly instead of polling output |
| 4 | TF-IDF `en-hi` / `en-mr` not flagged `degenerate` as the spec expects | Cosine is not exactly 0: scripts still share digits, punctuation and Latin acronyms, so `std < 1e-6` never fires | Extended the flag: degenerate also when `tau* <= 0` (all-positive predictions). This also flags `hi-mr`, which has real signal (AUC 0.605) but an all-positive F1 |
| 5 | Smoke run needed the model on CPU | Same RAM limit as #2 | Made the smoke set a subset of the full pairs so it reuses the full translation cache |
| 6 | Colab pip warning in red (diffusers/gradio vs huggingface-hub) | Unrelated pre-installed packages | Ignored; no effect on translation |
| 7 | Colab would not connect after the first session | Likely free GPU quota or account/browser issue (cause never confirmed) | Offered CPU runtime and Kaggle (CLI-driven) as fallbacks; Colab later connected, so not needed |
| 9 | Regex warning in `s05_diagnose.py` (capture group in a `str.contains` pattern) | Used a capturing group | Changed to a non-capturing group |
| 10 | Local Phase 6 encoding looked stalled (empty buffered log, swap about 3 GB) | Same 5 GB RAM limit; stdout was buffered so there was no progress signal | Killed it and ran on Colab. Also made `s06_ood.py` load models lazily so a cached re-run loads nothing |
| 11 | `pkill -f` / `pgrep -f` killed my own shell (exit 144) | The pattern matched the shell command line itself | Killed by exact PID via `ps` + `awk` instead |
| 12 | Stale partial OOD embeddings left locally after the kill | Killed run had finished IndicSBERT only | Deleted them so the Colab output is the single source |
| 8 | IndicTrans2 unusable | Gated repo, no HF token | NLLB fallback, as `CLAUDE.md` allows; licence recorded |

## Open items

- Re-run `s00_sample.py` from scratch to truly confirm identical pair IDs (currently only skip-if-exists).
- Kill-and-resume test of translation: skipped by choice.
- Git commit of all work: deferred by choice.
- Optional: IndicTrans2 path if a HF token and accepted terms become available.
- Final report must state: MT translationese, optimistic cross-lingual upper bound, hi-mr shared-script caveat, QQP label noise, NLLB non-commercial licence.
