"""Phase 8 step 4 (local): evaluate the fine-tuned embeddings with the SAME protocol/splits as the baseline,
write `indicsbert-ft` rows, and compare with frozen IndicSBERT (F1/AUC, B5-negative FPR, B1-positive recall)."""
import numpy as np
import pandas as pd

import config
from s04_evaluate import evaluate, write_metrics


def main():
    args = config.parse_args(__doc__)
    sfx = config.SMOKE_SUFFIX if args.smoke else ""
    d = pd.read_parquet(config.pairs_path(args.smoke))
    dev, test = (d.split == "dev").to_numpy(), (d.split == "test").to_numpy()
    y = d.label.to_numpy()
    E = {c: np.load(config.EMB_DIR / f"{config.FT_NAME}{sfx}_{c}.npy") for c in
         [f"q{i}_{l}" for l in config.LANGS for i in (1, 2)]}
    rows, scores = [], []
    for l1, l2 in config.CONDITIONS:
        s = (E[f"q1_{l1}"] * E[f"q2_{l2}"]).sum(1)
        r = evaluate(s[dev], y[dev], s[test], y[test])
        rows += [(l1, l2, "test", k, v) for k, v in r.items()]
        scores.append(pd.DataFrame({"pair_id": d.pair_id, "model": config.FT_NAME, "lang1": l1, "lang2": l2, "score": s}))
    write_metrics(config.FT_NAME, rows, args.smoke)
    sc = pd.concat(scores)
    sc.to_parquet(config.PROCESSED / f"scores_ft{sfx}.parquet", index=False)

    # OOD: tau on MahaParaphrase validation, report on test
    from datasets import load_dataset
    maha = load_dataset(config.MAHA)
    n_val, n_test = config.sizes(args.smoke)
    lab = lambda n, k: maha[n].to_pandas().head(k if args.smoke else None).Label.map(config.MAHA_LABELS).to_numpy()
    sv, st = [(lambda A, B: (A * B).sum(1))(*np.load(config.EMB_DIR / f"ood_{config.FT_NAME}{sfx}_{n}.npy")) for n in ("validation", "test")]
    r = evaluate(sv, lab("validation", n_val), st, lab("test", n_test))
    write_metrics(f"ood-{config.FT_NAME}", [("mr", "mr", "ood_test", k, v) for k, v in r.items()], args.smoke)

    # comparison vs frozen baseline (same dev/test, same protocol)
    m = pd.read_csv(config.METRICS_CSV.with_name("metrics_smoke.csv" if args.smoke else "metrics.csv"))
    g = lambda model, l1, l2, met: m[(m.model == model) & (m.lang1 == l1) & (m.lang2 == l2) & (m.metric == met)].value.iloc[0]
    base = pd.read_parquet(config.SCORES.with_name("scores_smoke.parquet" if args.smoke else "scores.parquet"))
    base = base[base.model == "indicsbert"]
    out = []
    for l1, l2 in config.CONDITIONS:
        row = {"condition": f"{l1}-{l2}"}
        for tag, model, S in (("base", "indicsbert", base), ("ft", config.FT_NAME, sc)):
            s = S[(S.lang1 == l1) & (S.lang2 == l2)].set_index("pair_id").score.loc[d.pair_id].to_numpy()
            pred = s >= g(model, l1, l2, "tau_star")
            b5, b1 = test & (d.bucket == 5).to_numpy() & (y == 0), test & (d.bucket == 1).to_numpy() & (y == 1)
            row |= {f"{tag}_f1": g(model, l1, l2, "f1"), f"{tag}_auc": g(model, l1, l2, "roc_auc"),
                    f"{tag}_B5neg_fpr": pred[b5].mean() if b5.any() else np.nan,
                    f"{tag}_B1pos_recall": pred[b1].mean() if b1.any() else np.nan}
        out.append(row)
    cmp = pd.DataFrame(out).round(3)
    cmp["d_auc"] = (cmp.ft_auc - cmp.base_auc).round(3)
    cmp.to_csv(config.RESULTS / f"ft_vs_base{sfx}.csv", index=False)
    print(cmp.to_string(index=False))
    print(f"OOD Marathi (ft): f1={r['f1']:.3f} auc={r['roc_auc']:.3f}  | baseline f1={g("ood-indicsbert","mr","mr","f1"):.3f} auc={g('ood-indicsbert','mr','mr','roc_auc'):.3f}")


if __name__ == "__main__":
    main()
