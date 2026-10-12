"""Phase 3: translate English questions to hi/mr, sqlite-cached by sha1(text + tgt_lang).

Backend: NLLB-200 (IndicTrans2 is gated; no HF token => documented fallback, CLAUDE.md §2.5/§5).
"""
import hashlib
import json
import re
import sqlite3

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def key(text, lang):
    return hashlib.sha1((text + lang).encode()).hexdigest()


def devanagari_frac(s):
    letters = [c for c in s if c.isalpha()]
    return sum(bool(DEVANAGARI.match(c)) for c in letters) / max(len(letters), 1)


def load_nllb():
    tok = AutoTokenizer.from_pretrained(config.NLLB_ID, src_lang=config.FLORES["en"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_ID, torch_dtype=dtype, low_cpu_mem_usage=True)
    return tok, model.to(dev).eval()


def translate(tok, model, texts, lang):
    bos = tok.convert_tokens_to_ids(config.FLORES[lang])
    assert bos != tok.unk_token_id, f"target token {config.FLORES[lang]} unresolved"
    b = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=config.TRANSLATE_MAXLEN).to(model.device)
    with torch.inference_mode():
        out = model.generate(**b, forced_bos_token_id=bos, max_length=config.TRANSLATE_MAXLEN,
                             num_beams=config.TRANSLATE_BEAMS)
    return tok.batch_decode(out, skip_special_tokens=True)


def main():
    args = config.parse_args(__doc__)
    path = config.pairs_path(args.smoke)
    d = pd.read_parquet(path)
    db = sqlite3.connect(config.TRANSLATION_CACHE)
    db.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, src TEXT, lang TEXT, out TEXT)")
    cached = {k for (k,) in db.execute("SELECT k FROM t")}
    texts = sorted(set(d.q1_en) | set(d.q2_en), key=len)  # length-sorted => less padding

    todo = {l: [t for t in texts if key(t, l) not in cached] for l in ("hi", "mr")}
    if any(todo.values()):
        tok, model = load_nllb()
        for l, ts in todo.items():
            if ts:  # sanity: 1 sentence must come back as Devanagari, not echoed English
                s = translate(tok, model, ts[:1], l)[0]
                assert devanagari_frac(s) > 0.5, f"{l}: not Devanagari: {s!r}"
                print(f"{l} check: {ts[0]!r} -> {s!r}")
            for i in range(0, len(ts), config.TRANSLATE_BATCH):
                batch = ts[i:i + config.TRANSLATE_BATCH]
                outs = translate(tok, model, batch, l)
                db.executemany("INSERT OR REPLACE INTO t VALUES (?,?,?,?)",
                               [(key(s, l), s, l, o) for s, o in zip(batch, outs)])
                db.commit()  # per-batch commit => kill-and-resume safe
                print(f"{l}: {min(i + len(batch), len(ts))}/{len(ts)}", flush=True)

    tr = {k: o for k, o in db.execute("SELECT k, out FROM t")}
    for l in ("hi", "mr"):
        for i in (1, 2):
            d[f"q{i}_{l}"] = d[f"q{i}_en"].map(lambda s: tr[key(s, l)]).astype("string")
    assert d[[f"q{i}_{l}" for i in (1, 2) for l in ("hi", "mr")]].notna().all().all()
    d.to_parquet(path, index=False)

    if not args.smoke:
        config.RESULTS.mkdir(exist_ok=True)
        config.RUN_CONFIG.write_text(json.dumps({
            "seed": config.SEED, "n_dev": config.N_DEV, "n_test": config.N_TEST,
            "encoders": config.MODELS, "translation_model": config.NLLB_ID,
            "translation_note": "IndicTrans2 gated/unaccepted -> NLLB fallback (CC-BY-NC-4.0)",
        }, indent=2))
    print(d[["q1_en", "q1_hi", "q1_mr"]].head(5).to_string())


if __name__ == "__main__":
    main()
