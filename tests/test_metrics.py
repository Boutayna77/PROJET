import pytest

from darija_nmt.metrics import degeneration_diagnostics, has_repeated_ngram, metric_tokenize, script_diagnostics


def test_metric_tokenize_splits_arabic_punct():
    assert metric_tokenize("كيداير؟") == "كيداير ؟"
    assert metric_tokenize("أنا، هنا.") == "أنا ، هنا ."


def test_script_and_repetition():
    d = script_diagnostics(["كيداير", "ok سلام", ""])
    assert abs(d["pure_arabic_script_rate"] - 1 / 3) < 1e-9 and abs(d["any_latin_rate"] - 1 / 3) < 1e-9
    assert has_repeated_ngram("a b c a b c".split(), 3) and not has_repeated_ngram("a b c d".split(), 3)
    g = degeneration_diagnostics(["هو هو هو هو", "انا هنا"], [["هو هنا"], ["انا هنا"]])
    assert g["stutter_rate"] == 0.5 and g["exact_match_rate"] == 0.5


def test_scores_multi_reference():
    pytest.importorskip("sacrebleu")
    from darija_nmt.metrics import corpus_scores
    hyps = ["انا كنشوف دار كبيرة هنا", "هو كيقرا كتاب"]
    refs = [["انا كنشوف دار كبيرة هنا"], ["شي حاجة اخرى تماما", "هو كيقرا كتاب"]]   # unequal reference counts
    s = corpus_scores(hyps, refs)
    assert abs(s["bleu"] - 100) < 1e-6 and abs(s["chrf"] - 100) < 1e-6
