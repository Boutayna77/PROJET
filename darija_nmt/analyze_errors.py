"""Error analysis from an existing predictions file: quality and degeneration by English sentence length.

  python -m darija_nmt.analyze_errors --pred runs/lstm/predictions_test_beam.csv
Writes a markdown table next to the predictions file (and prints it). Uses only text already produced by
evaluate.py, so it never touches the model.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .metrics import corpus_scores, has_repeated_ngram

BUCKETS = [(1, 3), (4, 6), (7, 9), (10, 10_000)]


def has_immediate_repeat(words: list[str]) -> bool:
    return any(a == b for a, b in zip(words, words[1:]))


def analyse(srcs, refs, hyps, min_n: int = 10):
    rows = []
    for lo, hi in BUCKETS + [(1, 10_000)]:
        idx = [i for i, s in enumerate(srcs) if lo <= len(s.split()) <= hi]
        if len(idx) < min_n:
            continue
        h, r = [hyps[i] for i in idx], [refs[i] for i in idx]
        sc = corpus_scores(h, r)
        n = len(idx)
        rows.append({
            "english_words": "all" if (lo, hi) == (1, 10_000) else (f"{lo}+" if hi >= 10_000 else f"{lo}-{hi}"),
            "sentences": n, "bleu": sc["bleu"], "chrf": sc["chrf"],
            "hyp_over_ref_words": sum(len(x.split()) for x in h) / max(sum(len(x[0].split()) for x in r), 1),
            "repeated_word_or_bigram_rate": sum(has_immediate_repeat(x.split()) or has_repeated_ngram(x.split(), 2) for x in h) / n,
            "exact_match_rate": sum(x.strip() in {y.strip() for y in rr} for x, rr in zip(h, r)) / n})
    return rows


def to_markdown(rows) -> str:
    keys = list(rows[0])
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--min-n", type=int, default=10, help="skip length buckets with fewer sentences")
    args = ap.parse_args(argv)
    srcs, refs, hyps = [], [], []
    with open(args.pred, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            srcs.append(r["src"]); refs.append(r["references"].split(" ||| ")); hyps.append(r["hypothesis"])
    table = to_markdown(analyse(srcs, refs, hyps, args.min_n))
    out = Path(args.pred).with_suffix(".by_length.md")
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
