"""Evaluate a trained run on val or test: loss, BLEU, chrF, script/degeneration diagnostics, examples.

Protocol: choose decoding settings (beam size, length penalty, no-repeat n-gram) on --split val,
then run --split test ONCE per model with the chosen settings.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from .checkpoint import load_run
from .dataset import ParallelDataset, load_eval_groups, make_loader
from .decoding import beam_translate, greedy_translate
from .metrics import bootstrap_ci, corpus_scores, degeneration_diagnostics, script_diagnostics, sentence_chrf
from .utils import evaluate_loss, get_device


def write_examples(path: Path, srcs, refs, hyps, seed=0, k=10):
    scored = sorted(((sentence_chrf(h, r), i) for i, (h, r) in enumerate(zip(hyps, refs))), key=lambda x: x[0])
    rnd = random.Random(seed).sample(range(len(srcs)), min(k, len(srcs)))

    def block(title, idxs):
        lines = [f"## {title}\n"]
        for i in idxs:
            lines += [f"- **EN:** {srcs[i]}", f"  - **REF:** {' | '.join(refs[i])}", f"  - **HYP:** {hyps[i]}",
                      f"  - sentence chrF: {sentence_chrf(hyps[i], refs[i]):.1f}\n"]
        return lines
    out = block("Most successful (highest sentence chrF)", [i for _, i in scored[::-1][:k]])
    out += block("Least successful (lowest sentence chrF)", [i for _, i in scored[:k]])
    out += block("Random sample (not cherry-picked)", rnd)
    path.write_text("\n".join(out), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--decode", default="both", choices=["greedy", "beam", "both"])
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=0.8)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--no-immediate-repeat", action="store_true", help="beam search never repeats a token twice in a row")
    ap.add_argument("--limit", type=int, default=None, help="evaluate decoding on a random subset of sources")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples for 95%% CIs (0=off)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    run = Path(args.run_dir)
    model, src_tok, tgt_tok, ckpt = load_run(run, device, args.checkpoint)
    csv_path = Path(args.data_dir) / f"{args.split}.csv"

    ds = ParallelDataset(csv_path, src_tok, tgt_tok)
    loss = evaluate_loss(model, make_loader(ds, 256, False, 0), device, 0)
    loss["nll_per_target_char"] = loss["nll_sum"] / max(sum(len(t) for t in ds.tgt_text), 1)
    print(f"{args.split} loss {loss['nll']:.4f} nats/token  ppl {loss['ppl']:.2f}  "
          f"{loss['nll_per_target_char']:.4f} nats/char  ({loss['tokens']} tokens)")

    srcs, refs = load_eval_groups(csv_path, limit=args.limit, seed=0)
    modes = ["greedy", "beam"] if args.decode == "both" else [args.decode]
    for mode in modes:
        t0 = time.time()
        if mode == "greedy":
            hyps = greedy_translate(model, src_tok, tgt_tok, srcs, device)
        else:
            hyps = []
            for n, s in enumerate(srcs, 1):
                hyps.append(beam_translate(model, src_tok, tgt_tok, s, device, beam_size=args.beam_size,
                                           length_penalty=args.length_penalty, no_repeat_ngram=args.no_repeat_ngram,
                                           block_immediate_repeat=args.no_immediate_repeat))
                if n % 200 == 0:
                    print(f"  beam {n}/{len(srcs)}  {(time.time() - t0) / n:.2f}s/sentence", flush=True)
        dt = time.time() - t0
        res = {"split": args.split, "mode": mode, "checkpoint_epoch": ckpt["epoch"], "n_sources": len(srcs),
               "loss": loss, **corpus_scores(hyps, refs), **script_diagnostics(hyps),
               **degeneration_diagnostics(hyps, refs), "decode_seconds_per_sentence": dt / max(len(srcs), 1),
               **(bootstrap_ci(hyps, refs, args.bootstrap) if args.bootstrap else {}),
               "decode_settings": {"beam_size": args.beam_size if mode == "beam" else 1,
                                   "length_penalty": args.length_penalty, "no_repeat_ngram": args.no_repeat_ngram if mode == "beam" else 0,
                                   "no_immediate_repeat": bool(args.no_immediate_repeat and mode == "beam")}}
        name = f"eval_{args.split}_{mode}{('_' + args.tag) if args.tag else ''}"
        (run / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(run / f"predictions_{args.split}_{mode}{('_' + args.tag) if args.tag else ''}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src", "references", "hypothesis"])
            for s, r, h in zip(srcs, refs, hyps):
                w.writerow([s, " ||| ".join(r), h])
        write_examples(run / f"examples_{args.split}_{mode}{('_' + args.tag) if args.tag else ''}.md", srcs, refs, hyps)
        print(f"[{mode}] BLEU {res['bleu']:.2f} chrF {res['chrf']:.2f} | pure-Arabic {res['pure_arabic_script_rate']:.3f} "
              f"| repeated-3gram {res['repeated_trigram_rate']:.3f} | len ratio {res['length_ratio_hyp_over_ref']:.2f} "
              f"| {res['decode_seconds_per_sentence']:.3f}s/sent")


if __name__ == "__main__":
    main()
