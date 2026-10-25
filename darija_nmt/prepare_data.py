"""Dataset inspection, cleaning, alignment checks and leakage-safe splitting.

Input : sentences.csv with columns eng, darija (Latin script), darija_ar (Arabic script)
Output: data/processed/{train,val,test}.csv (columns: id, src, tgt, group)
        data/processed/data_report.json   (every statistic reported in the paper)
        data/processed/dropped_samples.csv, audit_sample.csv (for manual inspection)

Nothing here is trained on: all thresholds are explicit CLI arguments and every filter's effect
is recorded in the funnel so the report can state exactly how many pairs each rule removed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .text_utils import (
    ARABIC_LETTER_RE, arabic_key, digit_set, english_key, has_url_or_email, n_words, normalize_darija_ar,
    normalize_english, script_class, script_counts,
)

BLANK_STRINGS = {"", "nan"}


def read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    cols = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in ("eng", "darija", "darija_ar") if c not in cols]
    if missing:
        raise SystemExit(f"Missing columns {missing}; found {list(df.columns)}")
    df = df.rename(columns={cols["eng"]: "eng", cols["darija"]: "darija", cols["darija_ar"]: "darija_ar"})
    df = df[["eng", "darija", "darija_ar"]].copy()
    for c in df.columns:
        df[c] = df[c].map(lambda s: "" if s.strip().lower() in BLANK_STRINGS else s)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def quantiles(values) -> dict:
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return {}
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 99, 100])
    return {"mean": float(v.mean()), **{k: float(x) for k, x in zip(["min", "p5", "p25", "p50", "p75", "p95", "p99", "max"], q)}}


def raw_statistics(df: pd.DataFrame) -> dict:
    n = len(df)
    s: dict = {"n_rows": n}
    for c in ("eng", "darija", "darija_ar"):
        blank = int((df[c] == "").sum())
        s[f"missing_{c}"] = {"count": blank, "rate": blank / n}
    both = (df.eng != "") & (df.darija_ar != "")
    s["rows_with_eng_and_darija_ar"] = int(both.sum())
    s["rows_with_darija_ar_but_no_eng"] = int(((df.eng == "") & (df.darija_ar != "")).sum())
    s["rows_with_eng_but_no_darija_ar"] = int(((df.eng != "") & (df.darija_ar == "")).sum())
    s["exact_duplicate_rows"] = int(df.duplicated(subset=["eng", "darija", "darija_ar"]).sum())
    nz = df[df.eng != ""]
    s["duplicate_english_strings"] = int(nz.duplicated(subset=["eng"]).sum())
    s["duplicate_eng_darija_ar_pairs"] = int(df[both].duplicated(subset=["eng", "darija_ar"]).sum())
    for c in ("eng", "darija", "darija_ar"):
        nonblank = df[df[c] != ""][c]
        counts = nonblank.map(script_class).value_counts().to_dict()
        s[f"script_distribution_{c}"] = {k: int(v) for k, v in counts.items()}
        s[f"words_per_sentence_{c}"] = quantiles(nonblank.map(n_words))
    return s


def run_filters(df: pd.DataFrame, args) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Apply ordered filters; return (clean df, funnel, dropped rows with reasons)."""
    df = df.copy()
    df["src"] = df.eng.map(normalize_english)
    df["tgt"] = df.darija_ar.map(lambda s: normalize_darija_ar(s, args.normalize_alef) if s else "")
    df["dar_latin"] = df.darija.map(normalize_english)
    df["reason"] = ""
    alive = pd.Series(True, index=df.index)
    funnel = [{"step": "raw_rows", "removed": 0, "remaining": int(len(df))}]

    def apply(name: str, mask: pd.Series):
        nonlocal alive
        hit = alive & mask
        df.loc[hit, "reason"] = name
        alive = alive & ~mask
        funnel.append({"step": name, "removed": int(hit.sum()), "remaining": int(alive.sum())})

    apply("missing_english", df.src == "")
    apply("missing_darija_arabic", df.tgt == "")

    src_ar = df.src.map(lambda t: script_counts(t)[0])
    apply("english_contains_arabic_letters", src_ar > 0)
    src_la = df.src.map(lambda t: script_counts(t)[1])
    apply("english_has_no_latin_letters", src_la == 0)

    # Chat-digit words in the ENGLISH column (e.g. "3yyeT 3lih") mean the row's columns are swapped.
    chat_word = re.compile(r"(?:^|\s)[2-9][A-Za-z]{3,}")
    apply("english_contains_chat_digit_word", df.src.map(lambda t: bool(chat_word.search(t))))

    tgt_counts = df.tgt.map(script_counts)
    tgt_ar = tgt_counts.map(lambda x: x[0])
    tgt_la = tgt_counts.map(lambda x: x[1])
    apply("target_has_no_arabic_letters", tgt_ar == 0)
    ratio = tgt_la / (tgt_la + tgt_ar).clip(lower=1)
    apply("target_latin_ratio_above_threshold", ratio > args.max_latin_ratio)

    apply("url_or_email", df.src.map(has_url_or_email) | df.tgt.map(has_url_or_email))

    sw, tw = df.src.map(n_words), df.tgt.map(n_words)
    apply("length_out_of_range",
          (sw < args.min_words) | (tw < args.min_words) | (sw > args.max_words) | (tw > args.max_words))

    # Alignment heuristic 1: robust character-length ratio outliers (only for non-tiny sentences).
    sc, tc = df.src.str.len(), df.tgt.str.len()
    logr = np.log((sc + 1) / (tc + 1))
    ref = logr[alive]
    med = float(ref.median()) if len(ref) else 0.0
    mad = float((ref - med).abs().median()) if len(ref) else 0.0
    scale = 1.4826 * mad if mad > 0 else 1.0
    big = np.maximum(sc, tc) >= args.ratio_min_chars
    apply("length_ratio_outlier", big & ((logr - med).abs() > args.ratio_mad_k * scale))

    # The converter that produced darija_ar maps chat digits to letters (2=ء 3=ع 7=ح ...) and so corrupts real
    # numbers ("10:30" -> "10:ع0"). Drop rows with English digits whose target has a token mixing digits and letters.
    def corrupted_digits(src: str, tgt: str) -> bool:
        return bool(re.search(r"\d", src)) and any(re.search(r"\d", w) and ARABIC_LETTER_RE.search(w) for w in tgt.split())
    apply("digit_letter_corruption_in_target",
          pd.Series([corrupted_digits(a, b) for a, b in zip(df.src, df.tgt)], index=df.index))

    # Alignment heuristic 2: digits present on both sides but disjoint.
    def digit_mismatch(row_src: str, row_tgt: str) -> bool:
        a, b = digit_set(row_src), digit_set(row_tgt)
        return bool(a) and bool(b) and a.isdisjoint(b)
    apply("digit_mismatch", pd.Series([digit_mismatch(a, b) for a, b in zip(df.src, df.tgt)], index=df.index))

    # Alignment heuristic 3: word count of Latin-script Darija vs Arabic-script Darija must agree.
    lw = df.dar_latin.map(n_words)
    cross = (lw >= 4) & (tw >= 4) & ((lw / tw.clip(lower=1)).between(1 / args.cross_ratio, args.cross_ratio) == False)  # noqa: E712
    apply("latin_vs_arabic_darija_wordcount_mismatch", cross)

    df["g"] = df.src.map(english_key)
    df["tk"] = df.tgt.map(arabic_key)
    apply("empty_key_after_normalisation", (df.g == "") | (df.tk == ""))

    dup = df.duplicated(subset=["g", "tk"], keep="first")
    apply("duplicate_pair_after_normalisation", dup)

    if args.max_variants > 0:
        rank = df[alive].groupby("g").cumcount()
        over = pd.Series(False, index=df.index)
        over.loc[rank.index] = rank >= args.max_variants
        apply("exceeds_max_variants_per_english", over)

    return df[alive].copy(), funnel, df[df.reason != ""][["row_id", "reason", "eng", "darija", "darija_ar"]]


def build_mono(raw: pd.DataFrame, args, exclude_keys: set[str]) -> list[str]:
    """Darija-only rows (no English) -> cleaned unique Arabic-script sentences, used ONLY as unlabeled text for the
    target-side subword vocabulary. Sentences equal to any val/test target are excluded."""
    out, seen = [], set()
    for t in raw[raw.eng == ""].darija_ar:
        t = normalize_darija_ar(t, args.normalize_alef) if t else ""
        if not t or has_url_or_email(t):
            continue
        ar, la = script_counts(t)
        if ar == 0 or la / (ar + la) > args.max_latin_ratio or not (args.min_words <= n_words(t) <= args.max_words):
            continue
        k = arabic_key(t)
        if k and k not in seen and k not in exclude_keys:
            seen.add(k)
            out.append(t)
    return out


def split_by_group(clean: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> dict[str, pd.DataFrame]:
    """Assign whole English-key groups to splits so no source sentence appears in two splits."""
    keys = sorted(clean.g.unique())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_val, n_test = int(round(n * val_frac)), int(round(n * test_frac))
    assign = {}
    for i, k in enumerate(keys):
        assign[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    out = {}
    for name in ("train", "val", "test"):
        part = clean[clean.g.map(assign) == name]
        out[name] = part
    return out


def split_summary(parts: dict[str, pd.DataFrame]) -> dict:
    res = {}
    train_targets = set(parts["train"].tk)
    for name, p in parts.items():
        d = {
            "pairs": int(len(p)),
            "unique_english": int(p.g.nunique()),
            "unique_targets": int(p.tk.nunique()),
            "src_words": quantiles(p.src.map(n_words)),
            "tgt_words": quantiles(p.tgt.map(n_words)),
            "pairs_per_english_mean": float(len(p) / max(p.g.nunique(), 1)),
        }
        if name != "train":
            d["target_string_also_in_train_rate"] = float(p.tk.isin(train_targets).mean()) if len(p) else 0.0
        res[name] = d
    a, b, c = (set(parts[k].g) for k in ("train", "val", "test"))
    res["english_key_overlap"] = {"train_val": len(a & b), "train_test": len(a & c), "val_test": len(b & c)}
    assert not (a & b) and not (a & c) and not (b & c), "source-side leakage between splits!"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw/sentences.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05, help="fraction of unique English sentences")
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--min-words", type=int, default=1)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--max-latin-ratio", type=float, default=0.0,
                    help="max fraction of Latin letters allowed in the Arabic-script target (0 = strict)")
    ap.add_argument("--ratio-mad-k", type=float, default=5.0, help="robust z-score cut for length ratio")
    ap.add_argument("--ratio-min-chars", type=int, default=12)
    ap.add_argument("--cross-ratio", type=float, default=2.5)
    ap.add_argument("--max-variants", type=int, default=0, help="cap Darija variants per English (0 = no cap)")
    ap.add_argument("--normalize-alef", action="store_true", help="unify أ إ آ -> ا in targets")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = read_raw(args.input)
    report: dict = {"args": vars(args), "raw": raw_statistics(raw)}

    clean, funnel, dropped = run_filters(raw, args)
    report["funnel"] = funnel
    report["clean_pairs"] = int(len(clean))
    if len(clean) < 100:
        raise SystemExit(f"Only {len(clean)} usable pairs after filtering - check thresholds / input.")

    # Diagnostics on the cleaned corpus.
    report["clean_alignment_diagnostics"] = {
        "english_ends_with_question_mark_but_target_has_no_arabic_qmark": float(
            ((clean.src.str.rstrip().str.endswith("?")) & (~clean.tgt.str.contains("؟"))).mean()),
        "pairs_per_english": quantiles(clean.groupby("g").size()),
    }

    parts = split_by_group(clean, args.val_frac, args.test_frac, args.seed)
    report["splits"] = split_summary(parts)

    for name, p in parts.items():
        ids = [f"{name}-{i}" for i in range(len(p))]
        pd.DataFrame({"id": ids, "src": p.src.values, "tgt": p.tgt.values, "group": p.g.values}).to_csv(
            out / f"{name}.csv", index=False, encoding="utf-8")

    # Samples for manual audit (alignment / naturalness) and for inspecting dropped rows.
    audit = clean.sample(n=min(100, len(clean)), random_state=args.seed)[["row_id", "src", "tgt", "darija"]]
    audit.assign(aligned_ok="", natural_ok="", notes="").to_csv(out / "audit_sample.csv", index=False, encoding="utf-8")
    samples = (pd.concat([g.sample(n=min(30, len(g)), random_state=args.seed) for _, g in dropped.groupby("reason")])
               if len(dropped) else dropped)
    samples.to_csv(out / "dropped_samples.csv", index=False, encoding="utf-8")

    mono = build_mono(raw, args, set(parts["val"].tk) | set(parts["test"].tk))
    (out / "mono_darija_ar.txt").write_text("\n".join(mono), encoding="utf-8")
    report["mono_darija_sentences_for_tokenizer"] = len(mono)

    (out / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    r = report["raw"]
    print(f"raw rows                          : {r['n_rows']}")
    print(f"rows with English                 : {r['n_rows'] - r['missing_eng']['count']}")
    print(f"rows with English AND Arabic Darija: {r['rows_with_eng_and_darija_ar']}")
    for f in funnel[1:]:
        print(f"  - {f['step']:<45} removed {f['removed']:>6}  remaining {f['remaining']:>6}")
    for k in ("train", "val", "test"):
        s = report["splits"][k]
        print(f"{k:<5}: {s['pairs']:>6} pairs / {s['unique_english']:>6} unique English sentences")
    print(f"Darija-only unlabeled sentences kept for the target vocabulary: {len(mono)}")
    print(f"Report written to {out / 'data_report.json'}")


if __name__ == "__main__":
    main()
