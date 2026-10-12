"""Phase 4 harness: threshold protocol + metric rows. Embedding-based evaluation lands in Phase 4."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

COLS = ["model", "lang1", "lang2", "split", "metric", "value"]
TAUS = np.arange(0, 1 + 1e-9, config.TAU_GRID_STEP)


def _prf(y, pred):
    """Positive-class accuracy/precision/recall/F1; vectorised over a leading axis."""
    tp, fp, fn = ((pred & (y == 1)).sum(-1), (pred & (y == 0)).sum(-1), (~pred & (y == 1)).sum(-1))
    p, r = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-12), (pred == (y == 1)).mean(-1)


def evaluate(s_dev, y_dev, s_test, y_test, n_boot=1000):
    """Sweep tau on dev, freeze, apply to test. Returns {metric: value} (test split)."""
    s_dev, s_test = np.asarray(s_dev, float), np.asarray(s_test, float)
    flat = s_dev.std() < 1e-6 or s_test.std() < 1e-6
    if flat:  # no signal: don't report a tuned tau*; predictions are all-positive
        tau = float("-inf")
    else:
        tau = TAUS[np.argmax([_prf(y_dev, s_dev >= t)[2] for t in TAUS])]
    # tau <= 0 also means all-positive (cosines tiny but nonzero, e.g. shared digits/punctuation)
    degenerate = flat or tau <= 0
    pred = s_test >= tau
    p, r, f1, acc = _prf(y_test, pred)
    idx = np.random.default_rng(config.SEED).integers(0, len(y_test), (n_boot, len(y_test)))
    boot = _prf(y_test[idx], pred[idx])[2]
    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "roc_auc": roc_auc_score(y_test, s_test) if not flat else 0.5,
        "tau_star": float("nan") if flat else tau,
        "f1_ci_low": np.percentile(boot, 2.5), "f1_ci_high": np.percentile(boot, 97.5),
        "degenerate": float(degenerate),
    }


def write_metrics(model, rows, smoke):
    """rows: [(lang1, lang2, split, metric, value)]. Replaces this model's rows, keeps others."""
    path = config.METRICS_CSV.with_name("metrics_smoke.csv") if smoke else config.METRICS_CSV
    new = pd.DataFrame([(model, *r) for r in rows], columns=COLS)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old.model != model], new])
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return path


def main():
    args = config.parse_args(__doc__)
    from s03_embed import emb_path

    d = pd.read_parquet(config.pairs_path(args.smoke))
    dev, test = (d.split == "dev").to_numpy(), (d.split == "test").to_numpy()
    y = d.label.to_numpy()
    m = evaluate(np.zeros(test.sum()), y[dev], np.zeros(test.sum()), y[test])  # constant => all-positive floor
    write_metrics("all-positive", [("*", "*", "test", k, v) for k, v in m.items()], args.smoke)

    scores = []
    for name in config.MODELS:
        E = {}
        for c in [f"q{i}_{l}" for l in config.LANGS for i in (1, 2)]:
            p = emb_path(name, c, args.smoke)
            if not p.exists():
                break
            E[c] = np.load(p)
            assert len(E[c]) == len(d), "embedding rows misaligned with pairs parquet"
        else:
            rows = []
            for l1, l2 in config.CONDITIONS:
                s = (E[f"q1_{l1}"] * E[f"q2_{l2}"]).sum(1)  # normalised => dot = cosine
                r = evaluate(s[dev], y[dev], s[test], y[test])
                rows += [(l1, l2, "test", k, v) for k, v in r.items()]
                scores.append(pd.DataFrame({"pair_id": d.pair_id, "model": name, "lang1": l1, "lang2": l2, "score": s}))
                print(f"{name:10s} {l1}-{l2} f1={r['f1']:.3f} auc={r['roc_auc']:.3f} tau={r['tau_star']:.2f} deg={r['degenerate']}")
            write_metrics(name, rows, args.smoke)
            continue
        print(f"{name}: embeddings missing, skipped")
    if scores:
        pd.concat(scores).to_parquet(config.SCORES.with_name("scores_smoke.parquet") if args.smoke else config.SCORES, index=False)


if __name__ == "__main__":
    main()
