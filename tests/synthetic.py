"""Tiny synthetic English/Darija corpus (word-for-word mapping) with injected noise.

Used ONLY for software tests. It is not Darija data and no result obtained on it means anything
about translation quality.
"""
from __future__ import annotations

import random

import pandas as pd

LEX = {
    "i": ("ana", "انا"), "you": ("nta", "نتا"), "we": ("7na", "حنا"), "he": ("huwa", "هو"),
    "she": ("hiya", "هي"), "see": ("kanchuf", "كنشوف"), "want": ("bghit", "بغيت"),
    "eat": ("kanakul", "كناكل"), "drink": ("kanchrab", "كنشرب"), "go": ("kanmchi", "كنمشي"),
    "read": ("kanqra", "كنقرا"), "write": ("kanktb", "كنكتب"), "big": ("kbir", "كبير"),
    "small": ("sghir", "صغير"), "house": ("dar", "دار"), "car": ("tomobil", "طوموبيل"),
    "book": ("ktab", "كتاب"), "water": ("ma", "ما"), "bread": ("khobz", "خبز"),
    "tea": ("atay", "اتاي"), "today": ("lyoum", "اليوم"), "tomorrow": ("ghda", "غدا"),
    "now": ("daba", "دابا"), "here": ("hna", "هنا"), "there": ("temma", "تما"),
    "friend": ("sahbi", "صاحبي"), "school": ("madrasa", "مدرسة"), "work": ("khdma", "خدمة"),
    "market": ("souq", "سوق"), "city": ("mdina", "مدينة"),
}


def make_corpus(n_pairs: int = 600, seed: int = 0, noisy: bool = True) -> pd.DataFrame:
    rng = random.Random(seed)
    words = list(LEX)
    rows, seen = [], set()
    while len(rows) < n_pairs:
        k = rng.randint(3, 7)
        ws = [rng.choice(words) for _ in range(k)]
        key = tuple(ws)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"eng": " ".join(ws).capitalize() + ".", "darija": " ".join(LEX[w][0] for w in ws),
                     "darija_ar": " ".join(LEX[w][1] for w in ws) + "."})
    df = pd.DataFrame(rows)
    if noisy:
        extra = []
        extra += [{"eng": "", "darija": "salam", "darija_ar": "سلام"}] * 20           # missing English
        extra += [{"eng": "Hello", "darija": "salam", "darija_ar": ""}] * 10           # missing Arabic
        extra += [{"eng": "I see the house", "darija": "kanchuf dar", "darija_ar": "I see dar"}] * 5  # Latin target
        extra += [{"eng": "I see a big house now", "darija": "ana kanchuf dar kbira",
                   "darija_ar": "انا كنشوف دار كبيرة دابا هههههه"}] * 1
        extra += [{"eng": "Visit www.example.com now", "darija": "x", "darija_ar": "زور www.example.com"}]
        extra += [df.iloc[0].to_dict()] * 3                                              # exact duplicates
        extra += [{"eng": df.iloc[1].eng.lower(), "darija": df.iloc[1].darija, "darija_ar": df.iloc[1].darija_ar}]
        extra += [{"eng": "I have 25 books at home today", "darija": "3ndi 25 ktab",
                   "darija_ar": "عندي 70 كتاب فالدار اليوم"}]                           # digit mismatch
        extra += [{"eng": "3yyeT 3lih", "darija": "call him", "darija_ar": "كال هيم"}]              # swapped columns
        extra += [{"eng": "It is at 10:30 today", "darija": "f 10:30", "darija_ar": "هادي 10:ع0 اليوم"}]  # corrupted digits
        extra += [{"eng": "He is very tall and kind and strong and happy",
                   "darija": "huwa twil bzaf", "darija_ar": "هو"}]                       # misaligned
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    make_corpus().to_csv(sys.argv[1], index=False, encoding="utf-8")
