import json

import pandas as pd

from darija_nmt.build_augmented import TAG, build
from darija_nmt.build_augmented import main as build_main
from darija_nmt.filter_synthetic import heuristic_filter
from darija_nmt.make_reverse import main as reverse_main
from darija_nmt.text_utils import arabic_key


def _write_split(d, split, rows):
    pd.DataFrame(rows, columns=["id", "src", "tgt", "group"]).to_csv(d / f"{split}.csv", index=False, encoding="utf-8")


def test_make_reverse_swaps_columns_and_groups_by_darija(tmp_path):
    data, tok = tmp_path / "p", tmp_path / "t"
    data.mkdir(); tok.mkdir()
    rows = [["train-0", "I see the house.", "انا كنشوف دار.", "i see the house"],
            ["train-1", "I look at the house.", "انا كنشوف دار.", "i look at the house"]]   # same Darija, 2 English
    for s in ("train", "val", "test"):
        _write_split(data, s, rows)
    (tok / "src.model").write_bytes(b"ENGLISH"); (tok / "tgt.model").write_bytes(b"DARIJA")
    reverse_main(["--data-dir", str(data), "--tok-dir", str(tok), "--out-data", str(tmp_path / "r"), "--out-tok", str(tmp_path / "rt")])
    rev = pd.read_csv(tmp_path / "r" / "train.csv", dtype=str, keep_default_na=False)
    assert rev.src.tolist() == ["انا كنشوف دار.", "انا كنشوف دار."] and rev.tgt.tolist()[0] == "I see the house."
    assert rev.group.nunique() == 1 and rev.group.iloc[0] == arabic_key("انا كنشوف دار.")   # both English are references of ONE source
    assert (tmp_path / "rt" / "src.model").read_bytes() == b"DARIJA" and (tmp_path / "rt" / "tgt.model").read_bytes() == b"ENGLISH"


def test_heuristic_filter_catches_bad_synthetic_pairs():
    df = pd.DataFrame({"src": ["I see the big house", "", "انا هنا", "see see the house", "go", "I see the big house",
                               "the same english", "the same english", "the same english", "the same english",
                               "He wants to eat bread now", "A B C A B C"],
                       "tgt": ["انا كنشوف دار كبيرة", "سلام", "انا هنا", "كنشوف دار هنا", "كنمشي دابا لدار ديالي هاد العام",
                               "كنشوف دار كبيرة هنا", "a1 b", "a2 b", "a3 b", "a4 b", "هو بغا ياكل خبز دابا", "ا ب ج"]})
    kept, funnel = heuristic_filter(df, forbidden_english_keys={"he wants to eat bread now"})
    reasons = {n: r for n, r, _ in funnel}
    assert reasons["empty_english"] == 1 and reasons["english_has_arabic_letters"] == 1
    assert reasons["length_ratio_outlier"] >= 1 and reasons["repeated_words_in_english"] >= 2
    assert reasons["english_equals_val_or_test_source"] == 1
    assert (kept.src == "the same english").sum() == 3            # capped at 3 variants
    assert "He wants to eat bread now" not in set(kept.src)


def test_build_augmented_tagging_mix_and_untouched_eval(tmp_path):
    real = pd.DataFrame({"id": [f"train-{i}" for i in range(10)], "src": [f"real {i}" for i in range(10)],
                         "tgt": [f"ر {i}" for i in range(10)], "group": [f"g{i}" for i in range(10)]})
    syn = pd.DataFrame({"src": [f"syn {i}" for i in range(30)], "tgt": [f"س {i}" for i in range(30)],
                        "chrf": [float(i) for i in range(30)]})
    mixed = build(real, syn, max_synthetic=20, real_upsample=2, tag=True, seed=1)
    assert len(mixed) == 2 * 10 + 20
    s = mixed[mixed.id.str.startswith("bt-")]
    assert len(s) == 20 and s.src.str.startswith(TAG).all() and not mixed[~mixed.id.str.startswith("bt-")].src.str.startswith(TAG).any()
    assert set(s.tgt) == {f"س {i}" for i in range(10, 30)}       # best-scored synthetic pairs are kept
    # CLI path: val/test copied byte-for-byte
    d = tmp_path / "p"; d.mkdir()
    real.to_csv(d / "train.csv", index=False, encoding="utf-8")
    real.head(3).to_csv(d / "val.csv", index=False, encoding="utf-8"); real.tail(3).to_csv(d / "test.csv", index=False, encoding="utf-8")
    syn.to_csv(tmp_path / "syn.csv", index=False, encoding="utf-8")
    build_main(["--data-dir", str(d), "--synthetic", str(tmp_path / "syn.csv"), "--out-dir", str(tmp_path / "o"), "--no-tag"])
    assert (tmp_path / "o" / "val.csv").read_bytes() == (d / "val.csv").read_bytes()
    assert (tmp_path / "o" / "test.csv").read_bytes() == (d / "test.csv").read_bytes()
    rep = json.loads((tmp_path / "o" / "build_report.json").read_text())
    assert rep["synthetic_pairs"] == 30 and rep["tagged"] is False
