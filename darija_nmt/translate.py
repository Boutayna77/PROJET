"""Inference interface: English sentence in, Moroccan Darija (Arabic script) out.

  python -m darija_nmt.translate --run-dirs runs/lstm --text "How are you?"
  python -m darija_nmt.translate --run-dirs runs/lstm runs/cifg runs/lstmp      # interactive, side by side

By default Arabic is reshaped/reordered so that terminals WITHOUT bidi support (macOS Terminal) display it
correctly. This is for display only; use --out to save the real (logical-order) text, or --raw to disable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import load_run
from .decoding import beam_translate, greedy_translate
from .text_utils import LATIN_LETTER_RE
from .utils import get_device


def make_display_fn(enabled: bool):
    """Return f(text)->text for terminal display. Needs `arabic-reshaper` and `python-bidi`."""
    if not enabled:
        return lambda t: t
    try:
        import arabic_reshaper
        try:
            from bidi import get_display          # python-bidi >= 0.5
        except ImportError:
            from bidi.algorithm import get_display  # python-bidi 0.4.x
    except ImportError:
        print("NOTE: Arabic is shown unshaped. Fix with:  pip install arabic-reshaper python-bidi")
        return lambda t: t
    return lambda t: get_display(arabic_reshaper.reshape(t))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--text", default=None, help="translate this sentence and exit (otherwise interactive)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=0.8)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-immediate-repeat", action="store_true", help="never repeat a token twice in a row")
    ap.add_argument("--raw", action="store_true", help="print the real text without terminal reshaping")
    ap.add_argument("--out", default=None, help="also append every translation (logical order) to this UTF-8 file")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    show = make_display_fn(not args.raw)
    loaded = [(Path(r).name, *load_run(r, device)[:3]) for r in args.run_dirs]

    def translate_all(text: str):
        for name, model, st, tt in loaded:
            if args.greedy:
                out = greedy_translate(model, st, tt, [text], device)[0]
            else:
                out = beam_translate(model, st, tt, text, device, beam_size=args.beam_size,
                                     length_penalty=args.length_penalty, no_repeat_ngram=args.no_repeat_ngram,
                                     block_immediate_repeat=args.no_immediate_repeat)
            flag = "   [warning: contains Latin letters]" if LATIN_LETTER_RE.search(out) else ""
            print(f"  [{name:<8}] {show(out)}{flag}")
            if args.out:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(f"{name}\t{text}\t{out}\n")

    if args.text is not None:
        print(f"EN: {args.text}")
        translate_all(args.text)
        return
    print("English -> Moroccan Darija (Arabic script). Empty line or Ctrl-D to quit.")
    while True:
        try:
            text = input("EN> ").strip()
        except EOFError:
            break
        if not text:
            break
        translate_all(text)


if __name__ == "__main__":
    main()
