"""Phase 8 step 2 (Colab GPU): translate training questions to hi/mr. Reuses s02's NLLB code and the sqlite cache."""
import sqlite3

import pandas as pd

import config
from s02_translate import key, load_nllb, translate


def main():
    args = config.parse_args(__doc__)
    path = config.TRAIN_PAIRS.with_name("train_pairs_smoke.parquet" if args.smoke else config.TRAIN_PAIRS.name)
    d = pd.read_parquet(path)
    db = sqlite3.connect(config.TRANSLATION_CACHE)
    db.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, src TEXT, lang TEXT, out TEXT)")
    cached = {k for (k,) in db.execute("SELECT k FROM t")}
    texts = sorted(set(d.q1_en) | set(d.q2_en), key=len)  # length-sorted => less padding
    todo = {l: [t for t in texts if key(t, l) not in cached] for l in ("hi", "mr")}
    if any(todo.values()):
        tok, model = load_nllb()
        for l, ts in todo.items():
            for i in range(0, len(ts), config.TRANSLATE_BATCH):
                batch = ts[i:i + config.TRANSLATE_BATCH]
                outs = translate(tok, model, batch, l)
                db.executemany("INSERT OR REPLACE INTO t VALUES (?,?,?,?)", [(key(s, l), s, l, o) for s, o in zip(batch, outs)])
                db.commit()  # per-batch commit => resumable after a disconnect
                if (i // config.TRANSLATE_BATCH) % 20 == 0:
                    print(f"{l}: {min(i + len(batch), len(ts))}/{len(ts)}", flush=True)
    tr = {k: o for k, o in db.execute("SELECT k, out FROM t")}
    for l in ("hi", "mr"):
        for i in (1, 2):
            d[f"q{i}_{l}"] = d[f"q{i}_en"].map(lambda s: tr[key(s, l)]).astype("string")
    assert d.notna().all().all()
    d.to_parquet(path, index=False)
    print("train pairs translated:", len(d))


if __name__ == "__main__":
    main()
