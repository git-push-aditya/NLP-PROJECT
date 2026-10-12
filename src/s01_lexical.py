"""Phase 2: char n-gram TF-IDF cosine baseline on every condition with translations present."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from s04_evaluate import evaluate, write_metrics


def main():
    args = config.parse_args(__doc__)
    d = pd.read_parquet(config.pairs_path(args.smoke))
    rows = []
    for l1, l2 in config.CONDITIONS:
        a, b = f"q1_{l1}", f"q2_{l2}"
        if d[a].isna().any() or d[b].isna().any():
            print(f"skip {l1}-{l2}: translations missing")
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([d[a], d[b]]))
        A, B = vec.transform(d[a]), vec.transform(d[b])
        s = np.asarray(A.multiply(B).sum(1)).ravel()  # rows are L2-normalised => dot = cosine
        dev, test = (d.split == "dev").to_numpy(), (d.split == "test").to_numpy()
        m = evaluate(s[dev], d.label[dev].to_numpy(), s[test], d.label[test].to_numpy())
        rows += [(l1, l2, "test", k, v) for k, v in m.items()]
        print(f"{l1}-{l2} f1={m['f1']:.3f} auc={m['roc_auc']:.3f} tau={m['tau_star']} degenerate={m['degenerate']}")
    print(write_metrics("tfidf", rows, args.smoke))


if __name__ == "__main__":
    main()
