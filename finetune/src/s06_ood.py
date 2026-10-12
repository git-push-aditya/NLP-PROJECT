"""Phase 6: out-of-domain control on human-written Marathi (MahaParaphrase). No training.

tau tuned on its own validation split, reported on its test split. Rows are named `ood-*` with
split `ood_test`, so they never mix with the headline table (and s04 never overwrites them).
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from s04_evaluate import evaluate, write_metrics


def main():
    args = config.parse_args(__doc__)
    n_val, n_test = config.sizes(args.smoke)
    ds = load_dataset(config.MAHA)
    sp = {}
    for name, n in (("validation", n_val), ("test", n_test)):
        df = ds[name].to_pandas().head(n if args.smoke else None)
        assert set(df.Label) <= set(config.MAHA_LABELS), "unexpected label strings"
        sp[name] = (df.Sentence1.tolist(), df.Sentence2.tolist(), df.Label.map(config.MAHA_LABELS).to_numpy())

    def run(name, score_fn):
        s = {k: score_fn(k, a, b) for k, (a, b, _) in sp.items()}
        m = evaluate(s["validation"], sp["validation"][2], s["test"], sp["test"][2])
        print(f"{name:10s} f1={m['f1']:.3f} auc={m['roc_auc']:.3f} tau={m['tau_star']:.2f} deg={m['degenerate']}")
        write_metrics(f"ood-{name}", [("mr", "mr", "ood_test", k, v) for k, v in m.items()], args.smoke)

    allt = [t for a, b, _ in sp.values() for t in a + b]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(allt)
    run("tfidf", lambda k, a, b: np.asarray(vec.transform(a).multiply(vec.transform(b)).sum(1)).ravel())
    run("all-positive", lambda k, a, b: np.zeros(len(a)))

    from sentence_transformers import SentenceTransformer
    for name, hf_id in config.MODELS.items():
        loaded = []  # lazy: a fully cached re-run never loads the model

        def score(k, a, b, name=name, hf_id=hf_id, loaded=loaded):
            p = config.EMB_DIR / f"ood_{name}{config.SMOKE_SUFFIX if args.smoke else ''}_{k}.npy"
            if not p.exists():
                if not loaded:
                    loaded.append(SentenceTransformer(hf_id))
                E = [loaded[0].encode(x, normalize_embeddings=True, batch_size=32, show_progress_bar=False) for x in (a, b)]
                np.save(p, np.stack(E))
            A, B = np.load(p)
            assert np.allclose(np.linalg.norm(A, axis=1), 1, atol=1e-3)
            return (A * B).sum(1)

        run(name, score)


if __name__ == "__main__":
    main()
