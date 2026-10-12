"""Demo: frozen IndicSBERT vs fine-tuned IndicSBERT, side by side. Each uses its own en-en tau* from metrics.csv."""
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent / "src"))
import config

t0 = time.time()
BASE = SentenceTransformer(config.MODELS["indicsbert"])
FT = SentenceTransformer(str(config.FT_MODEL_DIR)) if config.FT_MODEL_DIR.exists() else None
LOAD_SECONDS = time.time() - t0
m = pd.read_csv(config.METRICS_CSV)


def tau_of(model):
    q = m[(m.model == model) & (m.lang1 == "en") & (m.lang2 == "en") & (m.split == "test") & (m.metric == "tau_star")].value
    return float(q.iloc[0]) if len(q) else None


TAU_BASE, TAU_FT = tau_of("indicsbert"), tau_of(config.FT_NAME)
assert FT is None or TAU_FT is not None, "fine-tuned model found but no indicsbert-ft rows in metrics.csv: run s13_eval_ft.py"


def score(model, q1, q2):
    v = model.encode([q1, q2], normalize_embeddings=True)
    return round(float(v[0] @ v[1]), 4)


def predict(q1, q2):
    sb = score(BASE, q1, q2)
    out = [sb, "Duplicate" if sb >= TAU_BASE else "Not duplicate"]
    if FT is None:
        return out + [None, "(fine-tuned model not found)"]
    sf = score(FT, q1, q2)
    return out + [sf, "Duplicate" if sf >= TAU_FT else "Not duplicate"]


def seed_examples():
    """One real test pair per condition, from pairs the BASELINE gets right (texts differ, moderate margin); labels alternate.
    The fine-tuned model may disagree on some: that is shown honestly."""
    d = pd.read_parquet(config.PAIRS_ALL).query("split == 'test'")
    s = pd.read_parquet(config.SCORES).query("model == 'indicsbert'")
    ex, labels = [], []
    for k, (l1, l2) in enumerate(config.CONDITIONS):
        x = s[(s.lang1 == l1) & (s.lang2 == l2)].merge(d, on="pair_id")
        x = x[(x.score >= TAU_BASE) == (x.label == 1)]
        x = x[(x.label == (1 - k % 2)) & (x[f"q1_{l1}"].str.len() < 90) & (x[f"q2_{l2}"].str.len() < 90)]
        x = x[x[f"q1_{l1}"] != x[f"q2_{l2}"]]
        r = x.assign(gap=((x.score - TAU_BASE).abs() - 0.15).abs()).sort_values("gap").iloc[0]
        ex.append([r[f"q1_{l1}"], r[f"q2_{l2}"]])
        labels.append(f"{l1}-{l2} ({'duplicate' if r.label else 'not duplicate'} in QQP)")
    return ex, labels


EXAMPLES, LABELS = seed_examples()
th = lambda t: "n/a" if t is None else f"{t:.2f}"
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label="Question 1"), gr.Textbox(label="Question 2")],
    outputs=[gr.Number(label=f"Frozen IndicSBERT: cosine (tau*={th(TAU_BASE)})"), gr.Textbox(label="Frozen: prediction"),
             gr.Number(label=f"Fine-tuned IndicSBERT: cosine (tau*={th(TAU_FT)})"), gr.Textbox(label="Fine-tuned: prediction")],
    examples=EXAMPLES, example_labels=LABELS,
    title="Cross-lingual duplicate question detection (en / hi / mr): frozen vs fine-tuned",
    description="Same pair scored by both models, each with its own English-dev threshold. Examples are real held-out QQP test "
                "pairs (machine-translated for hi/mr) that the frozen model gets right.",
)

if __name__ == "__main__":
    print(f"models loaded in {LOAD_SECONDS:.1f}s")
    demo.launch()
