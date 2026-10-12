"""Phase 4a: encode every language column once per model, cache L2-normalised vectors as .npy."""
import numpy as np
import pandas as pd

import config


def emb_path(model, col, smoke):
    return config.EMB_DIR / f"{model}{config.SMOKE_SUFFIX if smoke else ''}_{col}.npy"


def main():
    args = config.parse_args(__doc__)
    from sentence_transformers import SentenceTransformer  # lazy: s04 needs no torch

    d = pd.read_parquet(config.pairs_path(args.smoke))
    config.EMB_DIR.mkdir(parents=True, exist_ok=True)
    cols = [f"q{i}_{l}" for l in config.LANGS for i in (1, 2)]
    for name, hf_id in config.MODELS.items():
        todo = [c for c in cols if not emb_path(name, c, args.smoke).exists()]
        if not todo:
            print(f"{name}: cached")
            continue
        model = SentenceTransformer(hf_id)
        for c in todo:
            e = model.encode(d[c].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-3), "embeddings not normalised"
            np.save(emb_path(name, c, args.smoke), e.astype("float32"))
            print(f"{name} {c} {e.shape}", flush=True)


if __name__ == "__main__":
    main()
