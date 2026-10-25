from darija_nmt.text_utils import (arabic_key, digit_set, english_key, latin_ratio, normalize_darija_ar,
                                   normalize_english, script_class)


def test_normalize_darija_basic():
    assert normalize_darija_ar("كِيْـــداير?") == "كيداير؟"          # diacritics, tatweel, ? -> ؟
    assert normalize_darija_ar("هههههه") == "هه"                      # elongation collapsed
    assert normalize_darija_ar("٣٠ درهم") == "30 درهم"                 # Arabic-Indic digits
    assert normalize_darija_ar("ک ی") == "ك ي"                        # Persian look-alikes
    assert normalize_darija_ar("  أنا   هنا ") == "أنا هنا"           # whitespace; alef kept by default
    assert normalize_darija_ar("أنا", normalize_alef=True) == "انا"


def test_normalize_english():
    assert normalize_english("  It’s  “fine” ") == "It's \"fine\""


def test_script_detection():
    assert script_class("كيداير") == "arabic"
    assert script_class("kidayr") == "latin"
    assert script_class("كيداير ok") == "mixed"
    assert script_class("123 !!") == "no_letters"
    assert abs(latin_ratio("ab ج") - 2 / 3) < 1e-9


def test_keys_and_digits():
    assert english_key("How are you?!") == english_key("how are  you")
    assert arabic_key("أنا هنا.") == arabic_key("انا هنا؟")
    assert digit_set("on 25 and 3") == {"25", "3"}
