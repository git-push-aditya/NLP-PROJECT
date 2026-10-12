# Cross-Lingual Duplicate Question Detection for Indian Languages

Context document for Claude Code. Read fully before writing any code.

---

## 0. What this project is

An academic NLP minor project (IIIT Kottayam, solo developer). The deliverable is a reproducible research pipeline that produces a comparison table, diagnostic plots, and a small demo — **not** a production service.

Grading weights what the code *demonstrates*, not what it deploys: literature grounding, baseline comparison, error analysis, and limitations. Optimise accordingly. A clean negative result is worth more than an unreproducible good number.

**Build an MVP that produces real numbers with zero model training.** Fine-tuning is a later phase and must not block the core result set.

---

## 1. Task definition

Given two questions `Q1` and `Q2`, written in the same or different languages, output a binary label:

```
s(Q1, Q2) -> [0, 1]        # similarity score
label = 1 if s >= tau else 0

```

`1` = Duplicate (same information need), `0` = Non-duplicate.

**Languages:** English (`en`), Hindi (`hi`), Marathi (`mr`). Chosen deliberately: `hi` and `mr` share the Devanagari script, `en` does not. This separates *script-crossing* transfer from *language-crossing* transfer.

**This is a pairwise classification task, not retrieval.** Two inputs, one label. Do not build a retrieval index, a vector DB, or a RAG pipeline. The task is `(Q1, Q2) -> {0,1}`.

### Evaluation matrix — six conditions, same underlying pairs


| Group                          | Conditions                | Question answered                               |
| ------------------------------ | ------------------------- | ----------------------------------------------- |
| Monolingual                    | `en-en`, `hi-hi`, `mr-mr` | Accuracy lost by leaving English                |
| Script-crossing                | `en-hi`, `en-mr`          | Does the shared space survive a script change   |
| Language-crossing, same script | `hi-mr`                   | Language transfer isolated from script transfer |


Every condition uses **identical pair IDs**, so F1 deltas are attributable to language alone. This is the central design constraint of the whole project — do not break it.

---

## 2. Grounded sources — pinned, do not substitute silently

### Anchor paper

**L3Cube-MahaParaphrase: A Marathi Paraphrase Detection Corpus and BERT-based Models** Jadhav, Shanbhag, Thakurdesai, Sinare, Joshi, Joshi. arXiv:2508.17444 (2025).

- 8,000 Marathi sentence pairs: 4,000 Paraphrase, 4,000 Non-paraphrase
- Verified by four native Marathi annotators
- Bucketed into five levels by word overlap: 0-20, 20-40, 40-60, 60-80, 80-100 %
- Word overlap defined as Jaccard: `|W(A) ∩ W(B)| / |W(A) ∪ W(B)|`
- Reported baselines (F1): MahaBERT **88.7**, IndicBERT (MLM+TLM) **87.1**, MuRIL **86.9**, IndicBERT (MLM only) **85.9**, mBERT **84.59**

**The gap this project fills:** that paper evaluates only *fine-tuned cross-encoders*, *monolingually*. It never tests whether a frozen multilingual embedding space already separates paraphrase from non-paraphrase, and never tests across languages.

### Models (HuggingFace IDs — use exactly these)


| Role                 | ID                                                            | Gated   |
| -------------------- | ------------------------------------------------------------- | ------- |
| Primary encoder      | `l3cube-pune/indic-sentence-similarity-sbert`                 | no      |
| Baseline encoder     | `sentence-transformers/LaBSE`                                 | no      |
| Baseline encoder     | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | no      |
| Translation          | `ai4bharat/indictrans2-en-indic-dist-200M`                    | **YES** |
| Translation fallback | `facebook/nllb-200-distilled-600M`                            | no      |


`indic-sentence-similarity-sbert` is fine-tuned from `l3cube-pune/indic-sentence-bert-nli` (MuRIL-based) on STS data across ten major Indian languages. Mean pooling. Loads directly with `SentenceTransformer(...)`, no custom code. Paper: Deode et al., *L3Cube-IndicSBERT*, arXiv:2304.11434, PACLIC 2023 pp. 154-163.

IndicTrans2 is MIT-licensed and covers all 22 scheduled Indian languages. Paper: Gala et al., TMLR 2023. **Its HuggingFace repo is gated** — see §2.5.

### Datasets


| Purpose                | Repo ID                      | Gated |
| ---------------------- | ---------------------------- | ----- |
| English base           | `nyu-mll/glue`, config `qqp` | no    |
| Gold Marathi OOD check | `l3cube-pune/MahaParaphrase` | no    |


Note the hyphen in `l3cube-pune`. The MahaParaphrase paper's own footnote writes the path as `l3cubepune/MahaParaphrase`, which **404s**. The organisation namespace is `l3cube-pune` for both models and datasets.

**QQP split sizes (verified):** `train` 363,846 labelled · `validation` 40,430 labelled · `test` 390,965 but **unlabelled** (`label == -1`). Total ~795k rows.

**Never use the GLUE** `test` **split.** It has no labels.

Carve dev and test from `validation`, not `train` — reduces the chance of overlap with data a pretrained encoder saw during its own fine-tuning.

---

## 2.5 Data acquisition

Everything downloads from the HuggingFace Hub into `~/.cache/huggingface`. Nothing is vendored into the repo; `data/raw/` holds only derived artifacts.

### Prerequisites

```bash
pip install "datasets>=2.18" "sentence-transformers>=3.0" "transformers>=4.51" \
            "torch>=2.5" "numpy>=2.1" pandas pyarrow scikit-learn matplotlib gradio
huggingface-cli login          # required for the gated translation model

# Phase 3 only, if using IndicTrans2 rather than the NLLB fallback:
pip install indictranstoolkit          # note: pypi name is all-lowercase

```

`transformers>=4.51`**,** `torch>=2.5`**,** `numpy>=2.1` **are floors set by IndicTransToolkit**, not by anything else here. Below them the toolkit does not guarantee stability. If Phase 3 takes the NLLB path these floors can be relaxed.

**IndicTransToolkit is Linux/macOS only and needs Python 3.10+.** It is explicitly not built or tested for Windows. On Windows, use WSL or go straight to the NLLB fallback.

`nyu-mll/glue` **is parquet-backed with per-config folders and no loading script** (verified), so it loads cleanly on `datasets` v3+ where script-based datasets were removed. No `trust_remote_code` needed.

### The one gate that will block an unattended run

`ai4bharat/indictrans2-en-indic-dist-200M` requires accepting terms ("You need to agree to share your contact information to access this model").

**A coding agent cannot click through this.** It must be done once, by a human, in a browser, before Phase 3:

1. Open [`https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
2. Log in, accept the conditions
3. `huggingface-cli login` with a token that has read access, or export `HF_TOKEN`

If this has not been done, the download fails with `401` / `GatedRepoError`. **Do not retry or work around it — fall through to the NLLB path in Phase 3.** `facebook/nllb-200-distilled-600M` is ungated and needs no token.

**Licence difference worth recording:** IndicTrans2 is MIT. NLLB-200 is **CC-BY-NC-4.0** (non-commercial). Fine for an academic project, but if the fallback is used, state the licence in the README — the generated Indic data inherits a non-commercial restriction.

### Download commands

```python
from datasets import load_dataset

# English base — ~795k rows across all splits, only `validation` is used
qqp = load_dataset("nyu-mll/glue", "qqp")

# Gold Marathi OOD control — 8,000 pairs
maha = load_dataset("l3cube-pune/MahaParaphrase")

```

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert")
SentenceTransformer("sentence-transformers/LaBSE")
SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

```

### Acquisition verification — run before Phase 1

Write this as `src/s_minus1_[verify.py](http://verify.py)` and run it first. It fails loudly rather than letting a wrong ID surface three phases later.

```python
assert set(qqp.keys()) == {"train", "validation", "test"}
assert qqp["train"].num_rows == 363_846
assert qqp["validation"].num_rows == 40_430
assert set(qqp["validation"]["label"]) == {0, 1}       # test split is all -1
assert set(qqp["validation"].column_names) >= {"question1", "question2", "label", "idx"}

```

**MahaParaphrase schema (verified — do not assume the usual conventions):**

```
columns:  Sentence1, Sentence2, Label      # capitalised, not sentence1/sentence2/label
Label:    "P" | "NP"                        # STRINGS, not 0/1
splits:   train ~5,000 · validation 999 · test ~2,000
total:    7,999 rows                        # not 8,000 — the paper rounds

```

Two consequences for Phase 6:

- **Map the label explicitly:** `{"P": 1, "NP": 0}`. A naive `int(label)` or an assumption of integer labels will throw or silently mislabel.
- **There is no bucket or overlap column**, despite the paper describing five buckets. Bucketing must be computed from `Sentence1` / `Sentence2`, not read from the dataset.

```python
assert maha["train"].column_names == ["Sentence1", "Sentence2", "Label"]
assert set(maha["train"]["Label"]) == {"P", "NP"}
assert sum(maha[s].num_rows for s in maha) == 7_999

```

QQP column names are expected to be `question1`, `question2`, `label`, `idx` — assert this on first run rather than trusting it.

### Offline / no-network environments

If the runtime has no outbound network, download on a machine that does and move the HF cache, or `snapshot_download` to a local path and point `load_dataset` / `SentenceTransformer` at that directory. Do not substitute a different dataset to work around a network failure.

---

## 3. Repository layout

```
dupq/
├── data/
│   ├── raw/                  # downloaded, never edited
│   ├── interim/              # splits, pair IDs
│   └── processed/            # pairs_all.parquet, translation cache
├── src/
│   ├── config.py             # all constants, seeds, paths, model IDs
│   ├── s_minus1_verify.py    # acquisition asserts — run first, see §2.5
│   ├── s00_sample.py
│   ├── s01_lexical.py
│   ├── s02_translate.py
│   ├── s03_embed.py
│   ├── s04_evaluate.py
│   ├── s05_diagnose.py
│   └── s06_ood.py
├── results/
│   ├── metrics.csv           # long format
│   └── figures/
├── app.py                    # gradio demo
├── report.ipynb              # imports from src/, renders results only
├── requirements.txt
└── README.md

```

**The notebook contains no logic.** It imports from `src/` and renders. Notebooks that hold the implementation are unreproducible and the project guidelines explicitly require reproducible documented code.

Every script is runnable standalone and idempotent: re-running a completed stage is a no-op that reads its cache.

---

## 4. Data contract

One canonical parquet, one row per pair. Every downstream stage reads this schema.

```
pair_id      int64      # stable, from QQP; the spine of the project
q1_en        string
q2_en        string
q1_hi        string
q2_hi        string
q1_mr        string
q2_mr        string
label        int8       # 1 = duplicate, 0 = non-duplicate
split        string     # "dev" | "test"
jaccard_en   float32    # token Jaccard on English side
bucket       int8       # 1..5 from jaccard_en

```

**Why** `jaccard_en` **is computed on English for all conditions:** in cross-script conditions raw overlap is trivially zero, which would collapse the buckets. The English source side keeps buckets comparable across all six conditions.

Bucket mapping: `B1` 0-20 %, `B2` 20-40 %, `B3` 40-60 %, `B4` 60-80 %, `B5` 80-100 %.

---

## 5. Phased implementation

Each phase has an acceptance criterion. **Do not start a phase until the previous one's criterion passes.** Phases 1-6 together are the MVP and require no GPU training.

---

### Phase 0 — Scaffold and smoke harness

**Goal:** the entire pipeline runs end-to-end on 50 pairs before anything scales.

- Create the repo layout above.
- `src/[config.py](http://config.py)` holds every constant: `SEED = 42`, paths, model IDs, `N_DEV`, `N_TEST`, `CONDITIONS`, bucket edges. Nothing hardcoded anywhere else.
- Add a `--smoke` flag to every script that caps `N_DEV = 30`, `N_TEST = 50`.
- Stub each stage to pass data through with the correct schema.

**Acceptance:** `make smoke` (or a shell script) runs stages 00→04 on 50 pairs and writes a `metrics.csv` with the right columns, even if the numbers are meaningless.

**This phase is non-negotiable.** Every integration bug surfaces here in two minutes instead of two hours at full scale.

---

### Phase 1 — Data acquisition and splits

**Goal:** frozen, balanced, reproducible splits with stable pair IDs.

```python
from datasets import load_dataset
ds = load_dataset("nyu-mll/glue", "qqp")   # use ds["validation"] only

```

- Sample from `validation` (40,430 labelled rows).
- **dev: 1,000 pairs · test: 2,000 pairs**, each balanced 50/50 on `label`. QQP's natural positive rate is ~37 %, so downsample negatives.
- Fix `seed=42`. Write pair IDs to `data/interim/pair_ids.json`.
- Compute `jaccard_en` and `bucket` now, on the English side.
- Write `data/processed/pairs_all.parquet` with the Indic columns empty.

**Why only 3,000 pairs — and what that precision actually buys.** With 2,000 balanced test pairs (1,000 positives), the standard error on F1 near 0.70 is roughly 1.5 %, so the **95 % confidence interval is about ±3 %**.

Practical consequence, which must be respected when reading the results table: **two methods differing by less than ~4 F1 points are not distinguishable.** Do not declare a winner on a 2-point gap. Report confidence intervals, not bare point estimates.

Cheapest way to get them: bootstrap the test set 1,000× from the **cached** score vectors (Phase 4 caches embeddings, so this costs seconds, not re-encoding). Emit `f1_ci_low` and `f1_ci_high` as their own metric rows.

Translation cost scales linearly with pair count and is the dominant time sink — that is the reason for 3,000 and not 25,000.

**Acceptance:** parquet exists, `label.mean() == 0.5` per split to within rounding, bucket distribution printed, re-running the script reproduces identical pair IDs.

---

### Phase 2 — Lexical baseline

**Goal:** a complete working result before any neural model is touched.

```python
TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

```

Character n-grams, not word n-grams — they degrade gracefully across scripts, which makes the baseline honest in cross-lingual conditions rather than trivially zero.

- Fit on the union of all questions in the relevant conditions.
- Score = cosine similarity.
- Sweep `tau` over `[0, 1]` in 0.01 steps **on dev**, pick argmax positive-class F1.
- Apply the frozen `tau*` to test.

### The degenerate-baseline trap — read before interpreting any cross-script number

Latin and Devanagari share **zero** character n-grams. So in `en-hi`, `en-mr` the TF-IDF cosine is exactly `0.0` for every pair. The threshold sweep then picks `tau <= 0`, the classifier predicts **all-positive**, and on a 50/50 balanced split that scores:

```
precision = 0.50   recall = 1.00   F1 = 0.667   ROC-AUC = 0.50

```

**F1 = 0.667 from a classifier with literally no signal.** That will sit in the results table looking like it beat a genuine cross-lingual model scoring 0.62. It is an artifact of F1 on balanced data, not a result.

Three required mitigations:

1. **Always report ROC-AUC next to F1.** AUC correctly reports 0.50 here. Any cell where F1 looks reasonable but AUC ≈ 0.50 is degenerate.
2. **Add an explicit** `all-positive` **row to** `metrics.csv` as a constant floor (F1 0.667, AUC 0.50). Any method not clearing it has no signal. This belongs in the report as a baseline, not a footnote.
3. **Guard the threshold sweep.** If `scores.std() < 1e-6`, do not report a tuned `tau*` — flag the cell `degenerate=True` and carry that flag into the table and plots.

The same trap applies to any condition where a model collapses. Check the flag, not the F1.

**Acceptance:** `metrics.csv` has a row for TF-IDF × `en-en` with a plausible F1 (expect roughly 0.60-0.70) and a recorded `tau*`; the `en-hi` and `en-mr` TF-IDF cells are flagged `degenerate=True`; the `all-positive` floor row exists.

---

### Phase 3 — Translation pipeline

**Goal:** fill `q*_hi` and `q*_mr`. This phase will consume more debugging time than any other. Treat it defensively.

**Volume:** 3,000 pairs × 2 questions × 2 target languages = **12,000 sentences**.

Model: `ai4bharat/indictrans2-en-indic-dist-200M` FLORES language codes: `eng_Latn` → `hin_Deva`, `mar_Deva`

**Required ordering:**

1. **Confirm the gate is cleared** (§2.5). If the repo 401s, do not debug it — the model is gated behind a manual terms acceptance that only a human in a browser can clear. Go straight to the fallback at step 3.
2. **Smoke-test on 20 sentences first.** Verify the output is actual Devanagari and not empty strings or echoed English.
3. IndicTrans2 needs `IndicTransToolkit` (github.com/VarunGumma/IndicTransToolkit, MIT) for its `IndicProcessor` wrapper, plus `trust_remote_code=True` on **both** the tokenizer and the model. Do not set `attn_implementation="flash_attention_2"` unless `flash_attn` is actually installed.

  **The processor is not optional.** Skipping `preprocess_batch` is the direct cause of the "empty or echoed English" failure:
  ```python
  from IndicTransToolkit import IndicProcessor          # Cython build
  # fallback if that import fails:
  # from IndicTransToolkit.IndicTransToolkit import IndicProcessor
  
  ip = IndicProcessor(inference=True)
  batch = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="hin_Deva")
  # ... tokenize -> model.generate(num_beams=5, max_length=256) -> batch_decode ...
  out = ip.postprocess_batch(decoded, lang="hin_Deva")
  
  ```

  The import path differs between the HF model card (`IndicTransToolkit.processor`) and the repo README (`IndicTransToolkit`). Try both; do not assume either.
4. **Fallback rule:** if the repo is gated-and-unaccepted, or `IndicTransToolkit` is not working within a reasonable debugging window, switch to `facebook/nllb-200-distilled-600M`. Ungated, same FLORES codes, works with vanilla `transformers`, no custom code path. Record the substitution in the README as a tooling constraint. A working NLLB pipeline beats a broken IndicTrans2 one; nothing downstream depends on which model produced the text.
5. The IndicTrans2 model card notes a newer "Indic-Translate" release that it recommends over IndicTrans2. Not evaluated here. If Phase 3 is being revisited, check it — but do not swap mid-project, and cite whatever is actually used.

**If using the NLLB fallback — API notes:**

The NLLB card warns that `pipeline("translation", ...)` is **removed in** `transformers` **v5**. Do not build on the pipeline API. Call `generate()` explicitly:

```python
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
bos = tok.convert_tokens_to_ids("hin_Deva")      # or "mar_Deva"
out = model.generate(**batch, forced_bos_token_id=bos, max_length=128, num_beams=5)

```

The `forced_bos_token_id` lookup has changed across `transformers` releases (`lang_code_to_id` was removed in newer versions). **Verify the target-language token resolves to a real id on 1 sentence before running 12,000** — a wrong id silently produces fluent output in the wrong language, which the adequacy audit will catch far too late.

**Caching is mandatory.** Key each translation on `sha1(source_text + tgt_lang)` and write to a sqlite or parquet cache after every batch. A runtime that dies at sentence 9,000 must resume, not restart.

**Batching:** sort by token length before batching to cut padding waste. `batch_size` 32-64, `max_length=128`, `num_beams=5`.

**Adequacy audit:** sample 60 translated pairs (20 per language pair), score each 1-3 for semantic adequacy manually, write to `results/adequacy_audit.csv`, report the mean. This is the project's measured noise floor and must be cited alongside every Indic number. Scale to 300 pairs before end-term.

**Acceptance:** all Indic columns non-null, 20 spot-checked translations are correct script and semantically plausible, cache survives a deliberate kill-and-resume test.

---

### Phase 4 — Embedding and evaluation harness

**Goal:** the 6 × 4 results table.

**The structural trick:** once each pair exists in all three languages, a condition is just a column pair.

```python
CONDITIONS = [("en","en"), ("hi","hi"), ("mr","mr"),
              ("en","hi"), ("en","mr"), ("hi","mr")]
# condition ("en","hi")  ->  q1_en  vs  q2_hi

```

**Encode each language column once per model and cache the matrix.** Six conditions then cost six dot products over cached vectors, not six encoding passes. Total encodings: 3 models × 6 columns × 3,000 rows.

```python
model.encode(texts, normalize_embeddings=True, batch_size=64)
# normalized -> cosine is a plain dot product

```

Threshold protocol, applied identically to every model and condition: sweep `tau` on **dev**, freeze, apply to **test**. Never tune on test.

**Metrics per cell:** accuracy, precision, recall, F1 (positive class), ROC-AUC on the raw score, and the selected `tau*`.

**Report** `tau*` **explicitly.** Drift in `tau*` across languages is itself a finding: it means the embedding space ranks correctly but is not calibrated across languages.

`metrics.csv` in long format so plotting is trivial:

```
model, lang1, lang2, split, metric, value

```

Everything is a metric row, including non-numeric signals — keeps the schema flat:

```
metric ∈ {accuracy, precision, recall, f1, roc_auc, tau_star,
          f1_ci_low, f1_ci_high, degenerate}      # degenerate: 0.0 | 1.0

```

**Acceptance:** 4 methods × 6 conditions × 6 metrics present in `metrics.csv`; the `en-en` row for IndicSBERT beats the TF-IDF `en-en` row on ROC-AUC.

---

### Phase 5 — Diagnostics and error analysis

**Goal:** the critical-analysis component. This is where marks are concentrated.

Break F1 down by `bucket` (from `jaccard_en`). Two slices carry the diagnostic weight:

- **B5 negatives** — 80-100 % overlap, `label == 0`. Precision here exposes lexical shortcutting. A model that scores well overall but fails here has learned overlap, not meaning.
- **B1 positives** — 0-20 % overlap, `label == 1`. Recall here is genuine semantic abstraction.

Additional slices: negation pairs, numeric/entity substitution, questions under 5 tokens.

Outputs:

- `results/figures/bucket_f1.png` — F1 by bucket, grouped by condition
- `results/figures/tau_drift.png` — `tau*` per condition per model
- `results/error_[examples.md](http://examples.md)` — 10 false positives and 10 false negatives with commentary

**Acceptance:** bucket plot exists; `error_[examples.md](http://examples.md)` contains real pairs pulled from test predictions with a one-line diagnosis each.

---

### Phase 6 — Out-of-domain check

**Goal:** prove the Indic results are not purely an artifact of machine translation.

```python
load_dataset("l3cube-pune/MahaParaphrase")   # hyphen — see §2.5

```

The corpus ships its own splits. Use them rather than re-splitting: tune `tau` on `validation` (999 rows), report on `test` (~2,000 rows). Ignore `train` entirely — see the no-training rule below.

Map labels first: `{"P": 1, "NP": 0}`. Columns are `Sentence1` / `Sentence2` / `Label`.

**Framing — state this explicitly in the report and the viva:** MahaParaphrase is **not training data**. It is a control. It is human-annotated, non-translated Marathi, so it controls for *translationese*. It is news-domain declarative sentences, not questions, so it does **not** control for question-ness.

**Do not train on it.** The domain shift from news declaratives to short interrogatives is real; a model fine-tuned on it will look good on Marathi and underperform on questions.

**Do not average its F1 into the headline table.** Separate row, separate table, labelled out-of-domain.

**Acceptance:** one clearly labelled OOD row in `metrics.csv`, excluded from headline aggregates.

---

### Phase 7 — Demo

```python
gr.Interface(fn=predict, inputs=[gr.Textbox(), gr.Textbox()], outputs=[...])

```

- Load the model once at module scope, not per request.
- Output both the raw cosine score and the binary label against the `en-en` `tau*`.
- **Pre-populate** `examples=[...]` **with one pair per condition.** In a viva you will not want to type Devanagari under time pressure.

**Acceptance:** demo runs locally, correctly classifies the six seeded examples.

---

### Phase 8 — Fine-tuning (deferred, end-term scope)

Do not start until Phases 1-7 are complete and committed.

- Fine-tune `l3cube-pune/indic-sentence-similarity-sbert` with `sentence-transformers`
- `OnlineContrastiveLoss` on mixed-language batches (`en-en`, `hi-hi`, `mr-mr`, `en-hi`), so the objective explicitly pulls cross-lingual duplicates together
- ~24,000 training pairs from QQP `train` (requires translating the training subset)
- 2 epochs, batch 32, `max_seq_length=64` (questions are short), AdamW 2e-5, 10 % warmup
- Optional Stage C: logistic regression over `[cos(u,v), |u-v|, u*v]` instead of a bare threshold, to test whether a single global `tau` is the limiting factor
- Optional: MuRIL fine-tuned as a cross-encoder — stronger but O(n²), non-deployable; mirrors the anchor paper's setup

---

## 6. Global engineering rules

- `SEED = 42` everywhere: `random`, `numpy`, `torch`, and every `train_test_split`.
- Every stage is idempotent and cache-backed. Re-running a completed stage is a no-op.
- Never tune a threshold on test. Dev only, then freeze.
- Pin versions in `requirements.txt`. Record the actual resolved versions in the README.
- Log every run's config to `results/run_config.json` — model IDs, seeds, split sizes, translation model actually used.
- No hardcoded paths outside [`config.py`](http://config.py).
- Prefer parquet over CSV for data, CSV only for `metrics.csv`.

---

## 7. Expected result bands — sanity check, not targets

Zero-shot cosine on QQP will **not** reach the 88.7 from the anchor paper. That number is a fine-tuned cross-encoder on a cleaner, balanced corpus. Realistic expectations:


| Condition                     | Expected F1                           | Expected AUC  |
| ----------------------------- | ------------------------------------- | ------------- |
| `en-en`, IndicSBERT zero-shot | 0.68 - 0.78                           | 0.75 - 0.85   |
| `hi-hi`, `mr-mr`              | several points below `en-en`          | below `en-en` |
| `en-hi`, `en-mr`              | lower again                           | lower again   |
| TF-IDF `en-en`                | 0.60 - 0.70, surprisingly competitive | 0.65 - 0.75   |
| TF-IDF cross-script           | **0.667 — degenerate, not a result**  | **0.50**      |
| `all-positive` floor          | 0.667                                 | 0.50          |


**Read the AUC column first.** On balanced data an all-positive classifier scores F1 0.667, so F1 alone cannot distinguish a weak model from no model at all. A cell only counts as a result if its AUC is meaningfully above 0.50.

**A mid-0.7 English number is the expected result, not a failure.** The contribution is the delta structure across six conditions and the bucket breakdown, not an absolute number. Do not panic-start fine-tuning on seeing 0.72 — that destroys the clean zero-shot baseline that makes a later fine-tuned number meaningful.

---

## 8. Known failure modes


| Symptom                                             | Likely cause                                                   | Action                                                                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| TF-IDF beats IndicSBERT on `en-en` F1               | `tau*` overfitting a 1,000-pair dev split                      | Check ROC-AUC. Higher AUC with lower F1 means ranking is fine and the threshold is the problem — that is the argument for the Stage C head. |
| Any method scores ~0.667 F1 with AUC ~0.50          | Degenerate all-positive prediction, not a result               | Check the `degenerate` flag and `scores.std()`. See Phase 2.                                                                                |
| MahaParaphrase load throws on labels                | Labels are strings `P`/`NP`, columns are capitalised           | Map `{"P":1,"NP":0}`; columns are `Sentence1`/`Sentence2`/`Label`                                                                           |
| NLLB output is fluent but the wrong language        | `forced_bos_token_id` resolved incorrectly                     | Verify the target token id on one sentence before the full run                                                                              |
| `GatedRepoError` / `401` on the translation model   | IndicTrans2 terms not accepted for this HF account             | Human accepts terms in a browser (§2.5), or fall back to ungated NLLB. Not debuggable by an agent.                                          |
| `404` / `RepositoryNotFoundError` on MahaParaphrase | Missing hyphen — `l3cubepune` instead of `l3cube-pune`         | Use `l3cube-pune/MahaParaphrase`                                                                                                            |
| Translation returns empty or echoed English         | `IndicProcessor` preprocessing not applied                     | Verify the processor wrapper is actually invoked; fall back to NLLB                                                                         |
| `hi-mr` scores suspiciously high                    | Shared Devanagari script and shared MuRIL tokenizer vocabulary | Expected and must be stated as a limitation. Malayalam is the listed extension that breaks the Devanagari assumption.                       |
| All cross-lingual F1 near 0.5                       | Language columns misaligned, or embeddings not normalized      | Assert `pair_id` alignment across columns; assert embedding norms ≈ 1                                                                       |
| MahaParaphrase F1 far above question F1             | Domain mismatch (news declaratives vs interrogatives)          | Expected. Keep it as a labelled OOD row. Do not merge into headline results.                                                                |


---

## 9. Known limitations to carry into the report

- QQP-Indic pairs are machine-translated and carry translationese and MT error. Mitigated, not removed, by the adequacy audit and the MahaParaphrase cross-check. **Never claim these are naturally occurring Indic questions.**
- **External validity of the cross-lingual conditions.** In `en-hi`, the comparison is an original English question against an MT rendering of its partner. That is a clean controlled test of *encoder alignment*, which is the stated research question — but real cross-lingual duplicates are independently authored, not translations of each other. MT output carries source-language structure, so it is plausibly **easier** to align to English than a natively written Hindi question would be. **The cross-lingual numbers should therefore be read as an optimistic upper bound**, not as deployment performance. State this explicitly; do not wait to be asked.
- QQP labels are crowd-sourced and contain known noise, capping achievable F1 even in English. The English number is a reference point, not a target.
- Hindi and Marathi share script and a large tokenizer vocabulary with MuRIL-based encoders, so `hi-mr` transfer is optimistic relative to a Dravidian-language pair.
- MahaParaphrase is news-domain sentences, not questions — validates paraphrase separability in Marathi, not question-duplicate performance.
- Single-developer project on free-tier GPU: model choice capped around 240M parameters, no hyper-parameter sweep beyond learning rate and epochs.

---

## 10. References

1. Jadhav, Shanbhag, Thakurdesai, Sinare, Joshi, Joshi. *MahaParaphrase: A Marathi Paraphrase Detection Corpus and BERT-based Models.* arXiv:2508.17444, 2025.
2. Deode, Gadre, Kajale, Joshi, Joshi. *L3Cube-IndicSBERT: A Simple Approach for Learning Cross-Lingual Sentence Representations using Multilingual BERT.* PACLIC 2023, pp. 154-163. arXiv:2304.11434.
3. Gala, Chitale, Raghavan, Gumma, Doddapaneni, et al. *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages.* TMLR, 2023.
4. Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP 2019, pp. 3982-3992.
5. Reimers, Gurevych. *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.* EMNLP 2020, pp. 4512-4525.
6. Feng, Yang, Cer, Arivazhagan, Wang. *Language-agnostic BERT Sentence Embedding (LaBSE).* ACL 2022, pp. 878-891.
7. Khanuja, Bansal, Mehtani, Khosla, Dey, et al. *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730, 2021.
8. Anand Kumar, Singh, Kavirajan, Soman. *Shared Task on Detecting Paraphrases in Indian Languages (DPIL): An Overview.* FIRE 2016, pp. 128-140.
9. Iyer, Dandekar, Csernai. *First Quora Dataset Release: Question Pairs.* Quora, 2017.

