"""Phase 5: bucket/slice breakdowns, tau drift plot, error examples. Reads cached scores + frozen tau*."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

MODELS = ["indicsbert", "labse", "mpnet", "tfidf"]
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # palette slots 1-4, fixed order
CONDS = [f"{a}-{b}" for a, b in config.CONDITIONS]
NEG = re.compile(r"\b(?:not|no|never|without|cannot|can.t|don.t|doesn.t|didn.t|isn.t|aren.t|won.t)\b", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")
DIAG = config.RESULTS / "error_diagnoses.json"  # hand-written one-line diagnoses, keyed by "FP:<pair_id>"/"FN:<pair_id>"


def load():
    d = pd.read_parquet(config.PAIRS_ALL)
    d = d[d.split == "test"].reset_index(drop=True)
    m = pd.read_csv(config.METRICS_CSV)
    tau = m[m.metric == "tau_star"].set_index(["model", "lang1", "lang2"]).value
    deg = m[m.metric == "degenerate"].set_index(["model", "lang1", "lang2"]).value
    s = pd.read_parquet(config.SCORES)
    return d, tau, deg, s


def tfidf_scores(d, l1, l2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    a, b = f"q1_{l1}", f"q2_{l2}"
    full = pd.read_parquet(config.PAIRS_ALL)  # same fit set as s01 (dev+test)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([full[a], full[b]]))
    return np.asarray(v.transform(d[a]).multiply(v.transform(d[b])).sum(1)).ravel()


def f1(y, p):
    tp, fp, fn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    config.parse_args(__doc__)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    d, tau, deg, s = load()
    y = d.label.to_numpy()

    # per-model per-condition test predictions
    pred, score = {}, {}
    for mod in MODELS:
        for l1, l2 in config.CONDITIONS:
            if mod == "tfidf":
                sc = tfidf_scores(d, l1, l2)
            else:
                q = s[(s.model == mod) & (s.lang1 == l1) & (s.lang2 == l2)].set_index("pair_id").score
                sc = q.loc[d.pair_id].to_numpy()
            t = tau[(mod, l1, l2)]
            score[mod, f"{l1}-{l2}"], pred[mod, f"{l1}-{l2}"] = sc, sc >= (t if not np.isnan(t) else -np.inf)

    # slices
    tok = lambda c: d[c].str.lower().str.findall(r"\w+")
    neg1, neg2 = d.q1_en.str.contains(NEG), d.q2_en.str.contains(NEG)
    n1, n2 = d.q1_en.map(lambda x: set(NUM.findall(x))), d.q2_en.map(lambda x: set(NUM.findall(x)))
    slices = {
        "B5_neg": (d.bucket == 5) & (y == 0), "B1_pos": (d.bucket == 1) & (y == 1),
        "negation_mismatch": neg1 ^ neg2,
        "number_mismatch": (n1 != n2) & ((n1.map(len) > 0) | (n2.map(len) > 0)),
        "short_lt5": (tok("q1_en").map(len) < 5) | (tok("q2_en").map(len) < 5),
    }
    rows = []
    for (mod, c), p in pred.items():
        for name, mask in slices.items():
            mk = mask.to_numpy()
            rows.append(dict(model=mod, condition=c, slice=name, n=int(mk.sum()), n_pos=int(y[mk].sum()),
                             accuracy=(p[mk] == (y[mk] == 1)).mean(),
                             fpr=p[mk & (y == 0)].mean() if (mk & (y == 0)).any() else np.nan,
                             recall=p[mk & (y == 1)].mean() if (mk & (y == 1)).any() else np.nan,
                             degenerate=bool(deg[(mod, *c.split("-"))])))
    pd.DataFrame(rows).round(3).to_csv(config.RESULTS / "slices.csv", index=False)

    # bucket F1 figure: one panel per condition, hollow dashed = degenerate (all-positive artefact)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, c in zip(axes.ravel(), CONDS):
        for mod in MODELS:
            f = [f1(y[d.bucket == b], pred[mod, c][d.bucket == b]) for b in range(1, 6)]
            bad = bool(deg[(mod, *c.split("-"))])
            ax.plot(range(1, 6), f, color=COLORS[mod], lw=2, ls="--" if bad else "-", marker="o", ms=5,
                    mfc="white" if bad else COLORS[mod], label=mod + (" (degenerate)" if bad else ""))
        ax.set_title(c, loc="left", fontsize=10)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(["B1\n0-20%", "B2", "B3", "B4", "B5\n80-100%"], fontsize=8)
        ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("F1 (positive class)"); axes[1, 0].set_ylabel("F1 (positive class)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:4], [x.replace(" (degenerate)", "") for x in l[:4]], ncol=4, loc="upper center", frameon=False)
    fig.text(.5, .01, "Bucket = word-overlap (Jaccard) of the English pair. Dashed/hollow = degenerate (all-positive) cell.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, .03, 1, .94)); fig.savefig(config.FIGURES / "bucket_f1.png", dpi=150); plt.close(fig)

    # tau drift: non-degenerate cells only
    fig, ax = plt.subplots(figsize=(7, 4))
    for mod in MODELS:
        pts = [(i, tau[(mod, *c.split("-"))]) for i, c in enumerate(CONDS) if not deg[(mod, *c.split("-"))]]
        ax.plot(*zip(*pts), color=COLORS[mod], lw=2, marker="o", ms=5, label=mod)
    ax.set_xticks(range(6)); ax.set_xticklabels(CONDS); ax.set_ylabel("tau* (tuned on dev)")
    ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.12))
    fig.tight_layout(); fig.savefig(config.FIGURES / "tau_drift.png", dpi=150); plt.close(fig)

    # error examples: IndicSBERT en-en, most confident errors
    p, sc = pred["indicsbert", "en-en"], score["indicsbert", "en-en"]
    diag = json.loads(DIAG.read_text()) if DIAG.exists() else {}
    out = ["# Error examples: IndicSBERT, en-en, test split",
           f"tau* = {tau[('indicsbert', 'en', 'en')]:.2f}. Most confident errors: false positives = highest-scoring "
           "non-duplicates; false negatives = lowest-scoring duplicates.\n"]
    for kind, mask, order in (("FP", p & (y == 0), False), ("FN", ~p & (y == 1), True)):
        idx = np.flatnonzero(mask); idx = idx[np.argsort(sc[idx])[: 10] if order else np.argsort(-sc[idx])[:10]]
        out.append(f"## {'False positives (predicted duplicate, actually not)' if kind == 'FP' else 'False negatives (predicted not duplicate, actually duplicate)'}\n")
        for k, i in enumerate(idx, 1):
            r = d.iloc[i]
            out += [f"**{kind}{k}** pair_id={r.pair_id} score={sc[i]:.3f} bucket=B{r.bucket}",
                    f"- Q1: {r.q1_en}", f"- Q2: {r.q2_en}",
                    f"- Diagnosis: {diag.get(f'{kind}:{r.pair_id}', 'TODO')}\n"]
    (config.RESULTS / "error_examples.md").write_text("\n".join(out))
    print(pd.DataFrame(rows).query("model != 'tfidf' and condition == 'en-en'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
