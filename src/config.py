"""All constants, seeds, paths, model IDs. Nothing is hardcoded elsewhere."""
import argparse
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW, INTERIM, PROCESSED = ROOT / "data/raw", ROOT / "data/interim", ROOT / "data/processed"
RESULTS, FIGURES = ROOT / "results", ROOT / "results/figures"
PAIRS_ALL = PROCESSED / "pairs_all.parquet"
PAIR_IDS = INTERIM / "pair_ids.json"
TRANSLATION_CACHE = PROCESSED / "translation_cache.sqlite"
EMB_DIR = PROCESSED / "emb"
SCORES = PROCESSED / "scores.parquet"
METRICS_CSV = RESULTS / "metrics.csv"
RUN_CONFIG = RESULTS / "run_config.json"
ADEQUACY_CSV = RESULTS / "adequacy_audit.csv"

N_DEV, N_TEST = 1000, 2000
SMOKE_N_DEV, SMOKE_N_TEST = 30, 50
SMOKE_SUFFIX = "_smoke"  # smoke artifacts never clobber full-run artifacts

LANGS = ["en", "hi", "mr"]
CONDITIONS = [("en", "en"), ("hi", "hi"), ("mr", "mr"), ("en", "hi"), ("en", "mr"), ("hi", "mr")]
FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}

BUCKET_EDGES = [0.2, 0.4, 0.6, 0.8]  # B1..B5
TAU_GRID_STEP = 0.01

QQP = ("nyu-mll/glue", "qqp")
MAHA = "l3cube-pune/MahaParaphrase"
MAHA_LABELS = {"P": 1, "NP": 0}
MODELS = {
    "indicsbert": "l3cube-pune/indic-sentence-similarity-sbert",
    "labse": "sentence-transformers/LaBSE",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
IT2_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
NLLB_ID = "facebook/nllb-200-distilled-600M"
TRANSLATE_BATCH, TRANSLATE_MAXLEN, TRANSLATE_BEAMS = 32, 128, 5


def parse_args(desc=""):
    """Shared CLI: every stage takes --smoke."""
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--smoke", action="store_true", help="cap to 30 dev / 50 test pairs")
    return p.parse_args()


def sizes(smoke):
    return (SMOKE_N_DEV, SMOKE_N_TEST) if smoke else (N_DEV, N_TEST)


def pairs_path(smoke):
    return PAIRS_ALL.with_name(f"pairs_all{SMOKE_SUFFIX}.parquet") if smoke else PAIRS_ALL
