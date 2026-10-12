"""Phase 8 step 1 (local): sample balanced training pairs from QQP `train`, leak-checked against dev/test."""
import pandas as pd
from datasets import load_dataset

import config


def main():
    args = config.parse_args(__doc__)
    n = 8 if args.smoke else config.N_TRAIN
    out = config.TRAIN_PAIRS.with_name("train_pairs_smoke.parquet" if args.smoke else config.TRAIN_PAIRS.name)
    if out.exists():
        print(f"{out.name} exists, skipping")
        return
    ev = pd.read_parquet(config.PAIRS_ALL)  # dev+test (3,000 pairs)
    seen = set(ev.q1_en) | set(ev.q2_en)
    df = load_dataset(*config.QQP)["train"].to_pandas()
    df = df[(df.question1.str.strip() != "") & (df.question2.str.strip() != "")]
    leak = df.question1.isin(seen) | df.question2.isin(seen)
    print(f"dropped {int(leak.sum())} train pairs sharing a question with dev/test")
    df = df[~leak]
    tr = pd.concat([df[df.label == l].sample(n // 2, random_state=config.SEED) for l in (0, 1)])
    tr = tr.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    res = pd.DataFrame({"pair_id": tr.idx.astype("int64"), "q1_en": tr.question1, "q2_en": tr.question2,
                        **{f"q{i}_{l}": pd.Series([None] * len(tr), dtype="string") for l in ("hi", "mr") for i in (1, 2)},
                        "label": tr.label.astype("int8")})
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    assert not (set(res.q1_en) | set(res.q2_en)) & seen, "train/eval leakage"
    print(res.label.mean(), len(res), "unique questions to translate:", len(set(res.q1_en) | set(res.q2_en)))


if __name__ == "__main__":
    main()
