"""Translation metrics and Arabic-script / degeneration diagnostics."""
from __future__ import annotations

import re

from .text_utils import ARABIC_LETTER_RE, LATIN_LETTER_RE

_PUNCT_SPLIT = re.compile(r"([.,!?؟،؛:;\"'()\[\]{}«»…])")


def metric_tokenize(s: str) -> str:
    """Whitespace + punctuation tokenisation that also splits Arabic punctuation (؟ ، ؛).
    The default sacreBLEU '13a' tokenizer ignores these, gluing them to the preceding word."""
    return re.sub(r"\s+", " ", _PUNCT_SPLIT.sub(r" \1 ", s)).strip()


def _ref_streams(refs_list: list[list[str]]) -> list[list[str]]:
    """Per-sentence reference lists (variable length) -> sacreBLEU reference streams.
    Short lists are padded by repeating their last reference; duplicates do not change BLEU/chrF."""
    n = max(len(r) for r in refs_list)
    return [[r[i] if i < len(r) else r[-1] for r in refs_list] for i in range(n)]


def corpus_scores(hyps: list[str], refs_list: list[list[str]]) -> dict:
    import sacrebleu
    bleu = sacrebleu.corpus_bleu([metric_tokenize(h) for h in hyps],
                                 _ref_streams([[metric_tokenize(x) for x in r] for r in refs_list]),
                                 tokenize="none")
    chrf = sacrebleu.corpus_chrf(hyps, _ref_streams(refs_list))
    return {"bleu": bleu.score, "chrf": chrf.score, "bleu_signature": f"sacreBLEU {sacrebleu.__version__}, tokenize=none (text pre-tokenised by metric_tokenize), smooth=exp",
            "bleu_brevity_penalty": bleu.bp, "bleu_sys_len": bleu.sys_len, "bleu_ref_len": bleu.ref_len}


def sentence_chrf(hyp: str, refs: list[str]) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, refs).score


def script_diagnostics(hyps: list[str]) -> dict:
    n = max(len(hyps), 1)
    has_latin = sum(1 for h in hyps if LATIN_LETTER_RE.search(h))
    all_arabic = sum(1 for h in hyps if ARABIC_LETTER_RE.search(h) and not LATIN_LETTER_RE.search(h))
    empty = sum(1 for h in hyps if not h.strip())
    lat_chars = sum(len(LATIN_LETTER_RE.findall(h)) for h in hyps)
    letters = sum(len(LATIN_LETTER_RE.findall(h)) + len(ARABIC_LETTER_RE.findall(h)) for h in hyps)
    return {"pure_arabic_script_rate": all_arabic / n, "any_latin_rate": has_latin / n,
            "latin_letter_share": lat_chars / max(letters, 1), "empty_rate": empty / n}


def has_repeated_ngram(tokens: list[str], n: int = 3) -> bool:
    seen = set()
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        if g in seen:
            return True
        seen.add(g)
    return False


def degeneration_diagnostics(hyps: list[str], refs_list: list[list[str]]) -> dict:
    n = max(len(hyps), 1)
    rep = sum(1 for h in hyps if has_repeated_ngram(h.split(), 3))
    # immediate word repetition such as "كيف كيف كيف"
    stutter = sum(1 for h in hyps if re.search(r"(?:^|\s)(\S+)(?:\s+\1){2,}(?:\s|$)", h))
    too_long = sum(1 for h, r in zip(hyps, refs_list) if len(h.split()) > 3 * max(len(r[0].split()), 1) + 5)
    hw = sum(len(h.split()) for h in hyps)
    rw = sum(len(r[0].split()) for r in refs_list)
    exact = sum(1 for h, r in zip(hyps, refs_list) if h.strip() in {x.strip() for x in r})
    return {"repeated_trigram_rate": rep / n, "stutter_rate": stutter / n, "overlong_rate": too_long / n,
            "length_ratio_hyp_over_ref": hw / max(rw, 1), "exact_match_rate": exact / n}


def bootstrap_ci(hyps: list[str], refs_list: list[list[str]], n: int = 200, seed: int = 0) -> dict:
    """95% percentile bootstrap CI over test sentences (the test set is small, so report uncertainty)."""
    import random
    rng = random.Random(seed)
    idx = range(len(hyps))
    bl, ch = [], []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        sc = corpus_scores([hyps[i] for i in pick], [refs_list[i] for i in pick])
        bl.append(sc["bleu"]); ch.append(sc["chrf"])
    q = lambda v: (sorted(v)[int(0.025 * len(v))], sorted(v)[min(int(0.975 * len(v)), len(v) - 1)])  # noqa: E731
    return {"bleu_ci95": q(bl), "chrf_ci95": q(ch), "bootstrap_resamples": n}
