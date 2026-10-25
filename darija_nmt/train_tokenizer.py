"""Train subword tokenizers on the TRAINING split only (no val/test leakage)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .tokenizer import SPTokenizer, train_spm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/tokenizer")
    ap.add_argument("--src-vocab", type=int, default=3000)
    ap.add_argument("--tgt-vocab", type=int, default=4000)
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--no-mono", action="store_true", help="do not add the Darija-only sentences to the target vocabulary text")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(args.data_dir) / "train.csv", dtype=str, keep_default_na=False)
    # unique sentences, so that sentences with many variants are not over-weighted in the vocabulary
    (out / "src.txt").write_text("\n".join(train.src.drop_duplicates()), encoding="utf-8")
    tgt_lines = list(train.tgt.drop_duplicates())
    mono_path = Path(args.data_dir) / "mono_darija_ar.txt"
    n_mono = 0
    if mono_path.exists() and not args.no_mono:  # unlabeled Darija (val/test targets already excluded by prepare_data)
        mono = [l for l in mono_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        tgt_lines += mono
        n_mono = len(mono)
    (out / "tgt.txt").write_text("\n".join(tgt_lines), encoding="utf-8")
    train_spm(str(out / "src.txt"), str(out / "src"), args.src_vocab, args.model_type, "nmt_nfkc")
    train_spm(str(out / "tgt.txt"), str(out / "tgt"), args.tgt_vocab, args.model_type, "identity")

    src_tok, tgt_tok = SPTokenizer(out / "src.model"), SPTokenizer(out / "tgt.model")
    report = {"args": vars(args), "mono_sentences_in_target_vocab_text": n_mono, "src_vocab_size": src_tok.vocab_size, "tgt_vocab_size": tgt_tok.vocab_size}
    for split in ("train", "val", "test"):
        df = pd.read_csv(Path(args.data_dir) / f"{split}.csv", dtype=str, keep_default_na=False)
        for side, tok in (("src", src_tok), ("tgt", tgt_tok)):
            ids = [tok.encode(t) for t in df[side]]
            n_tok = sum(len(i) for i in ids)
            n_words = sum(len(t.split()) for t in df[side])
            unk = sum(i.count(tok.unk_id) for i in ids)
            report[f"{split}_{side}"] = {"tokens_per_sentence": n_tok / max(len(ids), 1),
                                         "tokens_per_word": n_tok / max(n_words, 1),
                                         "unk_rate": unk / max(n_tok, 1),
                                         "max_len": max((len(i) for i in ids), default=0)}
    (out / "tokenizer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
