"""Translate unlabeled Darija sentences into (synthetic) English with a trained Darija->English run.

  python -m darija_nmt.backtranslate --reverse-run runs/rev_lstm
Output CSV columns: src (synthetic English), tgt (the ORIGINAL human-written Darija). Greedy decoding, batched.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from .checkpoint import load_run
from .decoding import greedy_translate
from .utils import get_device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reverse-run", required=True)
    ap.add_argument("--mono", default="data/processed/mono_darija_ar.txt")
    ap.add_argument("--out", default="data/synthetic/bt_raw.csv")
    ap.add_argument("--limit", type=int, default=None, help="random subset (for a quick test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    model, darija_tok, english_tok, _ = load_run(args.reverse_run, device)
    lines = [l.strip() for l in Path(args.mono).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit and args.limit < len(lines):
        lines = random.Random(args.seed).sample(lines, args.limit)
    syn, t0 = [], time.time()
    for i in range(0, len(lines), args.chunk):
        part = lines[i:i + args.chunk]
        syn += greedy_translate(model, darija_tok, english_tok, part, device, batch_size=args.batch_size)
        print(f"{min(i + args.chunk, len(lines))}/{len(lines)} translated  ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"src": syn, "tgt": lines}).to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {len(lines)} synthetic pairs to {args.out}")


if __name__ == "__main__":
    main()
