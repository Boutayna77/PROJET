"""Is `darija_ar` independent Arabic-script Darija, or a mechanical transliteration of the Latin `darija` column?

Test: align the two columns token by token (they have equal word counts) and measure how well the Latin word ALONE
determines the Arabic spelling. For independently written text one Latin spelling maps to many Arabic spellings.
"""
from __future__ import annotations

import argparse
import collections as C
import json
from pathlib import Path

import pandas as pd

from .text_utils import normalize_darija_ar


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/sentences.csv")
    ap.add_argument("--out", default="data/processed/transliteration_audit.json")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    same_len = int((df.darija.str.split().str.len() == df.darija_ar.str.split().str.len()).sum())
    pairs: dict[str, C.Counter] = C.defaultdict(C.Counter)
    n_tok = 0
    for lat, ar in zip(df.darija, df.darija_ar):
        lw, aw = lat.lower().split(), ar.split()
        if len(lw) != len(aw):
            continue
        for l, a in zip(lw, aw):
            pairs[l][normalize_darija_ar(a)] += 1
            n_tok += 1
    top = sum(max(c.values()) for c in pairs.values())
    freq = [(l, c) for l, c in pairs.items() if sum(c.values()) >= 5]
    single = sum(1 for _, c in freq if len(c) == 1)
    ambiguous = sorted(((sum(c.values()), l, dict(c.most_common(3))) for l, c in freq if len(c) > 1), reverse=True)[:15]
    res = {"rows": len(df), "rows_with_equal_word_count_in_both_columns": same_len,
           "aligned_tokens": n_tok, "latin_word_types": len(pairs),
           "share_of_tokens_explained_by_most_common_arabic_rendering": top / max(n_tok, 1),
           "frequent_types(>=5)": len(freq), "frequent_types_with_single_rendering": single,
           "frequent_types_with_single_rendering_rate": single / max(len(freq), 1),
           "most_frequent_ambiguous_words": ambiguous}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "most_frequent_ambiguous_words"}, indent=2))


if __name__ == "__main__":
    main()
