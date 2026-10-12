"""Phase 1: balanced dev/test splits from QQP validation, stable pair IDs, jaccard + bucket."""
import json
import re

import numpy as np
import pandas as pd
from datasets import load_dataset

import config

TOK = re.compile(r"\w+")


def jaccard(a, b):
    A, B = set(TOK.findall(a.lower())), set(TOK.findall(b.lower()))
    return len(A & B) / len(A | B) if A | B else 0.0


def main():
    args = config.parse_args(__doc__)
    n_dev, n_test = config.sizes(args.smoke)
    out = config.pairs_path(args.smoke)
    if out.exists():
        print(f"{out.name} exists, skipping")
        return
    if args.smoke and config.PAIRS_ALL.exists():  # reuse full pairs => translation cache hits, no model load
        full = pd.read_parquet(config.PAIRS_ALL)
        pd.concat([full[(full.split == sp) & (full.label == l)].head(n // 2)
                   for sp, n in (("dev", n_dev), ("test", n_test)) for l in (0, 1)]).to_parquet(out, index=False)
        print(f"smoke subset -> {out.name}")
        return
    df = load_dataset(*config.QQP)["validation"].to_pandas()
    df = df[(df.question1.str.strip() != "") & (df.question2.str.strip() != "")]
    parts = []
    for split, n in (("dev", n_dev), ("test", n_test)):  # sequential draw => disjoint splits
        pick = pd.concat(
            [df[df.label == l].sample(n // 2, random_state=config.SEED) for l in (0, 1)]
        )
        df = df.drop(pick.index)
        parts.append(pick.assign(split=split))
    d = pd.concat(parts).sort_values("idx").reset_index(drop=True)

    jac = np.array([jaccard(a, b) for a, b in zip(d.question1, d.question2)], dtype="float32")
    res = pd.DataFrame({
        "pair_id": d.idx.astype("int64"), "q1_en": d.question1, "q2_en": d.question2,
        **{f"q{i}_{l}": pd.Series([None] * len(d), dtype="string") for l in ("hi", "mr") for i in (1, 2)},
        "label": d.label.astype("int8"), "split": d.split,
        "jaccard_en": jac, "bucket": (np.digitize(jac, config.BUCKET_EDGES) + 1).astype("int8"),
    })[["pair_id", "q1_en", "q2_en", "q1_hi", "q2_hi", "q1_mr", "q2_mr",
        "label", "split", "jaccard_en", "bucket"]]
    config.PAIRS_ALL.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    if not args.smoke:
        ids = {s: sorted(res[res.split == s].pair_id.tolist()) for s in ("dev", "test")}
        config.PAIR_IDS.write_text(json.dumps(ids))
    print(res.groupby("split").label.mean())
    print(res.groupby(["split", "bucket"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
