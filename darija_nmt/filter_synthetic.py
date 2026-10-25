"""Quality filters for back-translated pairs (synthetic English -> real Darija).

1. Heuristic filters (no model): empty/Arabic-containing English, length-ratio outliers, repeated words,
   English equal to a val/test source (contamination), at most 3 Darija variants per synthetic English.
2. Optional round-trip filter: a forward English->Darija run (trained on REAL data only) re-translates the
   synthetic English; pairs whose output is close (sentence chrF) to the original Darija are kept.

  python -m darija_nmt.filter_synthetic --forward-run runs/lstm --keep-frac 0.5
  python -m darija_nmt.filter_synthetic --scored-in data/synthetic/bt_scored.csv --keep-frac 0.25   # no model needed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import has_repeated_ngram
from .text_utils import english_key, script_counts


def _repeated(words: list[str]) -> bool:
    return any(a == b for a, b in zip(words, words[1:])) or has_repeated_ngram(words, 2)


def heuristic_filter(df: pd.DataFrame, forbidden_english_keys: set[str], min_ratio: float = 0.5,
                     max_ratio: float = 3.5, max_variants: int = 3):
    """df has columns src (synthetic English) and tgt (Darija). Returns (kept dataframe, funnel list)."""
    df = df.copy()
    df["src"] = df.src.fillna("").astype(str).str.strip()
    df["key"] = df.src.map(english_key)
    alive = pd.Series(True, index=df.index)
    funnel = [("input", 0, int(len(df)))]

    def apply(name, mask):
        nonlocal alive
        hit = alive & mask
        alive = alive & ~mask
        funnel.append((name, int(hit.sum()), int(alive.sum())))

    apply("empty_english", df.src == "")
    counts = df.src.map(script_counts)
    apply("english_has_arabic_letters", counts.map(lambda c: c[0] > 0))
    apply("english_has_no_latin_letters", counts.map(lambda c: c[1] == 0))
    ratio = df.src.str.split().str.len() / df.tgt.str.split().str.len().clip(lower=1)
    apply("length_ratio_outlier", (ratio < min_ratio) | (ratio > max_ratio))
    apply("repeated_words_in_english", df.src.map(lambda s: _repeated(s.lower().split())))
    apply("english_equals_val_or_test_source", df.key.isin(forbidden_english_keys))
    kept = df[alive].copy()
    rank = kept.groupby("key").cumcount()
    funnel.append((f"more_than_{max_variants}_variants_per_english", int((rank >= max_variants).sum()),
                   int((rank < max_variants).sum())))
    kept = kept[rank < max_variants]
    return kept.drop(columns=["key"]).reset_index(drop=True), funnel


def roundtrip_scores(df: pd.DataFrame, forward_run: str, device_name: str, batch_size: int = 64, chunk: int = 4000):
    from .checkpoint import load_run
    from .decoding import greedy_translate
    from .metrics import sentence_chrf
    from .utils import get_device
    device = get_device(device_name)
    model, src_tok, tgt_tok, _ = load_run(forward_run, device)
    hyps = []
    for i in range(0, len(df), chunk):
        hyps += greedy_translate(model, src_tok, tgt_tok, df.src.iloc[i:i + chunk].tolist(), device, batch_size)
        print(f"round-trip {min(i + chunk, len(df))}/{len(df)}", flush=True)
    return [sentence_chrf(h, [t]) for h, t in zip(hyps, df.tgt)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/synthetic/bt_raw.csv")
    ap.add_argument("--data-dir", default="data/processed", help="for val/test English keys (contamination check)")
    ap.add_argument("--forward-run", default=None, help="forward run trained on real data only (round-trip filter)")
    ap.add_argument("--scored-in", default=None, help="reuse an earlier bt_scored.csv instead of recomputing")
    ap.add_argument("--scored-out", default="data/synthetic/bt_scored.csv")
    ap.add_argument("--out", default="data/synthetic/bt_filtered.csv")
    ap.add_argument("--keep-frac", type=float, default=0.5, help="keep the best fraction by round-trip chrF")
    ap.add_argument("--min-chrf", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    if args.scored_in:
        scored = pd.read_csv(args.scored_in, dtype={"src": str, "tgt": str}, keep_default_na=False)
        scored["chrf"] = scored.chrf.astype(float)
    else:
        raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        forbidden = set()
        for split in ("val", "test"):
            forbidden |= set(pd.read_csv(Path(args.data_dir) / f"{split}.csv", dtype=str, keep_default_na=False).group)
        scored, funnel = heuristic_filter(raw, forbidden)
        for name, removed, remaining in funnel:
            print(f"  {name:<45} removed {removed:>6}  remaining {remaining:>6}")
        if args.forward_run:
            scored["chrf"] = roundtrip_scores(scored, args.forward_run, args.device)
            Path(args.scored_out).parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.scored_out, index=False, encoding="utf-8")
            print(f"scored pairs saved to {args.scored_out}")
    if "chrf" in scored.columns:
        print("round-trip chrF quantiles (10/25/50/75/90):",
              [round(float(scored.chrf.quantile(q)), 1) for q in (.1, .25, .5, .75, .9)])
        cut = max(float(scored.chrf.quantile(1 - args.keep_frac)), args.min_chrf)
        scored = scored[scored.chrf >= cut]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False, encoding="utf-8")
    print(f"kept {len(scored)} synthetic pairs -> {args.out}")


if __name__ == "__main__":
    main()
