"""Text normalisation and script-detection utilities (pure Python, no heavy dependencies)."""
from __future__ import annotations

import re
import unicodedata

# --- character classes ---------------------------------------------------------------------
ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u06FA-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F]")
_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")
_AR_REPEAT_RE = re.compile(r"([\u0621-\u064A])\1{2,}")  # 3+ repeated Arabic letters -> 2
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(https?://|www\.|\S+@\S+\.\S+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")

_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Persian / Urdu look-alike letters that occasionally leak into Arabic-script Darija text.
_CHAR_MAP = str.maketrans({"ی": "ي", "ک": "ك", "ۀ": "ه", "ە": "ه"})
_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_AR_PUNCT_MAP = str.maketrans({"?": "؟", ",": "،", ";": "؛"})
_EN_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"'})


def normalize_english(text: str) -> str:
    """NFKC, strip invisible characters, straighten quotes, collapse whitespace."""
    t = unicodedata.normalize("NFKC", str(text))
    t = _INVISIBLE_RE.sub("", t).translate(_EN_QUOTE_MAP)
    return _WS_RE.sub(" ", t).strip()


def normalize_darija_ar(text: str, normalize_alef: bool = False) -> str:
    """Light normalisation of Arabic-script Darija.

    Removes diacritics/tatweel/invisible marks, maps Persian look-alike letters, converts
    Arabic-Indic digits to ASCII, unifies ? , ; to Arabic punctuation, collapses letter
    elongation (3+ repeats -> 2). Alef/hamza unification is optional because it changes
    legitimate spelling.
    """
    t = unicodedata.normalize("NFKC", str(text))
    t = _INVISIBLE_RE.sub("", t)
    t = _DIACRITICS_RE.sub("", t).replace("\u0640", "")
    t = t.translate(_DIGIT_MAP).translate(_CHAR_MAP)
    if normalize_alef:
        t = t.translate(_ALEF_MAP)
    t = t.translate(_AR_PUNCT_MAP)
    t = _AR_REPEAT_RE.sub(r"\1\1", t)
    return _WS_RE.sub(" ", t).strip()


def english_key(text: str) -> str:
    """Case/punctuation-insensitive key used for de-duplication and split grouping."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\u00c0-\u024f ]+", " ", t)
    return _WS_RE.sub(" ", t).strip()


def arabic_key(text: str) -> str:
    """Key for de-duplicating Arabic-script targets (ignores punctuation and alef variants)."""
    t = normalize_darija_ar(text, normalize_alef=True)
    t = re.sub(r"[^\u0621-\u064A\u066E-\u06D3\u06FA-\u06FF\u0750-\u077F\u08A0-\u08FF0-9 ]+", " ", t)
    return _WS_RE.sub(" ", t).strip()


# --- script detection ---------------------------------------------------------------------
def script_counts(text: str) -> tuple[int, int]:
    """(# Arabic letters, # Latin letters)."""
    return len(ARABIC_LETTER_RE.findall(text)), len(LATIN_LETTER_RE.findall(text))


def latin_ratio(text: str) -> float:
    ar, la = script_counts(text)
    return la / (ar + la) if (ar + la) else 0.0


def arabic_ratio(text: str) -> float:
    ar, la = script_counts(text)
    return ar / (ar + la) if (ar + la) else 0.0


def script_class(text: str) -> str:
    """One of: no_letters, arabic, latin, mixed."""
    ar, la = script_counts(text)
    if ar + la == 0:
        return "no_letters"
    if la == 0:
        return "arabic"
    if ar == 0:
        return "latin"
    return "mixed"


def has_url_or_email(text: str) -> bool:
    return bool(_URL_RE.search(text))


def digit_set(text: str) -> set[str]:
    return set(_DIGITS_RE.findall(text.translate(_DIGIT_MAP)))


def n_words(text: str) -> int:
    return len(text.split())
