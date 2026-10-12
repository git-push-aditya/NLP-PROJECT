"""Phase 8 step 3 (Colab GPU): fine-tune IndicSBERT with OnlineContrastiveLoss on mixed-language pairs, then
encode dev/test (6 columns) and MahaParaphrase (val/test) with the tuned model. Evaluation happens locally (s13)."""
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from transformers import set_seed

import config


def main():
    args = config.parse_args(__doc__)
    sfx = config.SMOKE_SUFFIX if args.smoke else ""
    set_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    tp = config.TRAIN_PAIRS.with_name("train_pairs_smoke.parquet" if args.smoke else config.TRAIN_PAIRS.name)
    tr = pd.read_parquet(tp)
    assert tr.notna().all().all(), "train pairs not fully translated (run s11 first)"
    # each pair is shown in ONE randomly chosen language condition => mixed-language batches
    conds = [config.FT_CONDITIONS[i] for i in rng.integers(0, len(config.FT_CONDITIONS), len(tr))]
    ds = Dataset.from_dict({
        "sentence1": [r[f"q1_{a}"] for (a, b), (_, r) in zip(conds, tr.iterrows())],
        "sentence2": [r[f"q2_{b}"] for (a, b), (_, r) in zip(conds, tr.iterrows())],
        "label": tr.label.astype(int).tolist()})

    model = SentenceTransformer(config.MODELS["indicsbert"])
    model.max_seq_length = config.FT_MAXLEN
    cuda = torch.cuda.is_available()
    targs = SentenceTransformerTrainingArguments(
        output_dir="/tmp/ft_out", num_train_epochs=1 if args.smoke else config.FT_EPOCHS,
        per_device_train_batch_size=8 if args.smoke else config.FT_BATCH, learning_rate=config.FT_LR,
        warmup_ratio=config.FT_WARMUP, fp16=cuda, seed=config.SEED, save_strategy="no", report_to="none",
        logging_steps=50, use_cpu=not cuda)
    t0 = time.time()
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds,
                                         loss=losses.OnlineContrastiveLoss(model))
    trainer.train()
    train_s = time.time() - t0
    model.max_seq_length = 128  # eval-time length, same as the frozen baseline

    enc = lambda x: model.encode(x, normalize_embeddings=True, batch_size=64, show_progress_bar=False).astype("float32")
    config.EMB_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(config.pairs_path(args.smoke))
    for l in config.LANGS:
        for i in (1, 2):
            np.save(config.EMB_DIR / f"{config.FT_NAME}{sfx}_q{i}_{l}.npy", enc(d[f"q{i}_{l}"].tolist()))
    maha = load_dataset(config.MAHA)
    n_val, n_test = config.sizes(args.smoke)
    for name, n in (("validation", n_val), ("test", n_test)):
        df = maha[name].to_pandas().head(n if args.smoke else None)
        np.save(config.EMB_DIR / f"ood_{config.FT_NAME}{sfx}_{name}.npy", np.stack([enc(df.Sentence1.tolist()), enc(df.Sentence2.tolist())]))
    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / f"ft_run{sfx}.json").write_text(json.dumps({
        "train_pairs": len(tr), "epochs": targs.num_train_epochs, "batch": targs.per_device_train_batch_size,
        "lr": config.FT_LR, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"done: train {train_s:.0f}s on {'GPU' if cuda else 'CPU'}")


if __name__ == "__main__":
    main()
