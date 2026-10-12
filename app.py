"""Phase 7 demo: duplicate-question check with IndicSBERT, thresholded at the en-en tau*."""
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent / "src"))
import config

t0 = time.time()
MODEL = SentenceTransformer(config.MODELS["indicsbert"])  # once, at module scope
LOAD_SECONDS = time.time() - t0
m = pd.read_csv(config.METRICS_CSV)
TAU = float(m[(m.model == "indicsbert") & (m.lang1 == "en") & (m.lang2 == "en")
              & (m.split == "test") & (m.metric == "tau_star")].value.iloc[0])


def predict(q1, q2):
    v = MODEL.encode([q1, q2], normalize_embeddings=True)
    score = float(v[0] @ v[1])
    return round(score, 4), "Duplicate" if score >= TAU else "Not duplicate", TAU


def seed_examples():
    """One real test pair per condition: correct, texts differ, moderate margin (not trivially easy); labels alternate."""
    d = pd.read_parquet(config.PAIRS_ALL).query("split == 'test'")
    s = pd.read_parquet(config.SCORES).query("model == 'indicsbert'")
    ex, labels = [], []
    for k, (l1, l2) in enumerate(config.CONDITIONS):
        x = s[(s.lang1 == l1) & (s.lang2 == l2)].merge(d, on="pair_id")
        x = x[(x.score >= TAU) == (x.label == 1)]
        x = x[(x.label == (1 - k % 2)) & (x[f"q1_{l1}"].str.len() < 90) & (x[f"q2_{l2}"].str.len() < 90)]
        x = x[x[f"q1_{l1}"] != x[f"q2_{l2}"]]
        r = x.assign(gap=((x.score - TAU).abs() - 0.15).abs()).sort_values("gap").iloc[0]
        ex.append([r[f"q1_{l1}"], r[f"q2_{l2}"]])
        labels.append(f"{l1}-{l2} ({'duplicate' if r.label else 'not duplicate'} in QQP)")
    return ex, labels


EXAMPLES, LABELS = seed_examples()

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label="Question 1"), gr.Textbox(label="Question 2")],
    outputs=[gr.Number(label="Cosine score"), gr.Textbox(label="Prediction"),
             gr.Number(label="Threshold (en-en tau*)")],
    examples=EXAMPLES, example_labels=LABELS,
    title="Cross-lingual duplicate question detection (en / hi / mr)",
    description="IndicSBERT, frozen, zero-shot. Same-language or cross-language pairs; threshold tuned on English dev. "
                "Examples are real held-out QQP test pairs (machine-translated for hi/mr), one per condition.",
)

if __name__ == "__main__":
    print(f"model loaded in {LOAD_SECONDS:.1f}s, tau*={TAU:.2f}")
    demo.launch()
