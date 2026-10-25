import json

import pandas as pd

from darija_nmt.prepare_data import main as prepare
from tests.synthetic import make_corpus


def test_prepare_pipeline(tmp_path):
    raw = tmp_path / "sentences.csv"
    make_corpus(300, seed=1).to_csv(raw, index=False, encoding="utf-8")
    out = tmp_path / "processed"
    prepare(["--input", str(raw), "--out-dir", str(out), "--seed", "1"])
    rep = json.loads((out / "data_report.json").read_text(encoding="utf-8"))
    reasons = {f["step"]: f["removed"] for f in rep["funnel"]}
    # every injected defect class must be caught
    assert reasons["missing_english"] == 20
    assert reasons["missing_darija_arabic"] == 10
    assert reasons["target_has_no_arabic_letters"] == 5
    assert reasons["target_latin_ratio_above_threshold"] >= 1
    assert reasons["english_contains_chat_digit_word"] == 1
    assert reasons["digit_letter_corruption_in_target"] == 1
    assert reasons["digit_mismatch"] == 1
    assert reasons["length_ratio_outlier"] >= 1
    assert reasons["duplicate_pair_after_normalisation"] >= 3
    train, val, test = (pd.read_csv(out / f"{s}.csv", dtype=str, keep_default_na=False) for s in ("train", "val", "test"))
    assert len(train) + len(val) + len(test) == rep["clean_pairs"]
    # no source sentence is shared between splits; no Latin letters in targets
    g = [set(d.group) for d in (train, val, test)]
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
    assert not train.tgt.str.contains("[A-Za-z]").any()
    mono = (out / "mono_darija_ar.txt").read_text(encoding="utf-8").splitlines()
    assert mono == ["سلام"] and rep["mono_darija_sentences_for_tokenizer"] == 1
    assert rep["splits"]["english_key_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
