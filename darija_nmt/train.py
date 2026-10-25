"""Train one English->Darija model. Example:
    python -m darija_nmt.train --cell lstm --run-name lstm
Checkpoint selection uses VALIDATION loss only; the test set is never touched here.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .dataset import ParallelDataset, load_eval_groups, make_loader
from .decoding import greedy_translate
from .metrics import corpus_scores
from .model import ModelConfig, Seq2Seq
from .tokenizer import SPTokenizer
from .utils import build_optimizer, evaluate_loss, get_device, seq_loss, set_seed

LOG_FIELDS = ["epoch", "lr", "train_loss_dropout", "train_loss_eval", "val_loss", "val_ppl",
              "val_bleu_greedy", "val_chrf_greedy", "epoch_seconds", "total_seconds",
              "mean_grad_norm", "skipped_steps"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tok-dir", default="data/tokenizer")
    # model
    ap.add_argument("--cell", default="lstm", choices=["lstm", "cifg", "lstmp", "wmc"])
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--proj-size", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--no-tie-weights", action="store_true")
    ap.add_argument("--init", default="orthogonal", choices=["orthogonal", "xavier", "uniform"])
    ap.add_argument("--forget-bias", type=float, default=1.0)
    # optimisation
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "rmsprop", "sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    ap.add_argument("--lr-patience", type=int, default=1, help="ReduceLROnPlateau patience")
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    # data / misc
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train-pairs", type=int, default=None, help="random subset (for tuning/benchmarks)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="cut epochs short (benchmark/smoke)")
    ap.add_argument("--bleu-every", type=int, default=1, help="greedy BLEU on a val subset every N epochs (0=off)")
    ap.add_argument("--bleu-sentences", type=int, default=500)
    ap.add_argument("--train-eval-sentences", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="load model weights from this checkpoint (warm start / fine-tuning)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = Path(args.tok_dir)
    (run_dir / "tokenizer").mkdir(exist_ok=True)
    for f in ("src.model", "tgt.model"):
        shutil.copy(tok_dir / f, run_dir / "tokenizer" / f)
    src_tok, tgt_tok = SPTokenizer(run_dir / "tokenizer" / "src.model"), SPTokenizer(run_dir / "tokenizer" / "tgt.model")

    data = Path(args.data_dir)
    train_ds = ParallelDataset(data / "train.csv", src_tok, tgt_tok, args.max_len, limit=args.max_train_pairs, seed=args.seed)
    val_ds = ParallelDataset(data / "val.csv", src_tok, tgt_tok, args.max_len)
    train_loader = make_loader(train_ds, args.batch_size, True, 0, args.seed)
    val_loader = make_loader(val_ds, 256, False, 0)
    train_eval_loader = make_loader(train_ds.take(args.train_eval_sentences, args.seed), 256, False, 0)
    bleu_srcs, bleu_refs = load_eval_groups(data / "val.csv", limit=args.bleu_sentences, seed=args.seed)

    cfg = ModelConfig(cell=args.cell, emb_dim=args.emb_dim, hidden_size=args.hidden_size, proj_size=args.proj_size,
                      layers=args.layers, dropout=args.dropout, attention=not args.no_attention,
                      tie_weights=not args.no_tie_weights, init=args.init, forget_bias=args.forget_bias)
    model = Seq2Seq(cfg, src_tok.vocab_size, tgt_tok.vocab_size, pad_id=0).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=False)["model"])
        print(f"initialised weights from {args.init_from}")
    params = model.param_breakdown()
    opt = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=args.lr_factor, patience=args.lr_patience)

    config_record = {"args": vars(args), "model_cfg": cfg.to_dict(), "params": params, "device": str(device),
                     "n_train_pairs": len(train_ds), "n_val_pairs": len(val_ds),
                     "n_truncated_train": train_ds.n_truncated, "n_truncated_val": val_ds.n_truncated,
                     "src_vocab": src_tok.vocab_size, "tgt_vocab": tgt_tok.vocab_size,
                     "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    print(f"[{args.run_name}] cell={args.cell} device={device} trainable params={params['total']:,} "
          f"(recurrent cells: {params['recurrent_cells_total']:,}) train pairs={len(train_ds):,}")

    start_epoch, best_val, bad_epochs, elapsed_before = 1, float("inf"), 0, 0.0
    log_path = run_dir / "log.csv"
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best_val, bad_epochs, elapsed_before = ck["epoch"] + 1, ck["best_val"], ck["bad_epochs"], ck["elapsed"]
        print(f"resumed from epoch {ck['epoch']} (best val {best_val:.4f})")
    else:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()

    best_epoch, stopped_early = None, False
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loader.batch_sampler.set_epoch(epoch)
        run_nll = torch.zeros((), device=device)
        run_tok = torch.zeros((), device=device)
        gn_sum, gn_n, skipped = 0.0, 0, 0
        for step, (src, tgt) in enumerate(train_loader, 1):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            loss, nll_sum, ntok = seq_loss(logits, tgt[:, 1:], 0, args.label_smoothing)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(gn):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            run_nll += nll_sum
            run_tok += ntok
            gn_sum += float(gn)
            gn_n += 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"train nll/token {run_nll.item() / max(run_tok.item(), 1):.4f}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        train_loss_dropout = run_nll.item() / max(run_tok.item(), 1)

        val = evaluate_loss(model, val_loader, device, 0)
        tr_eval = evaluate_loss(model, train_eval_loader, device, 0)
        bleu = chrf = float("nan")
        if args.bleu_every and epoch % args.bleu_every == 0:
            hyps = greedy_translate(model, src_tok, tgt_tok, bleu_srcs, device)
            sc = corpus_scores(hyps, bleu_refs)
            bleu, chrf = sc["bleu"], sc["chrf"]

        sched.step(val["nll"])
        epoch_s = time.time() - t0
        total_s = elapsed_before + (time.time() - t_start)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss_dropout": train_loss_dropout,
               "train_loss_eval": tr_eval["nll"], "val_loss": val["nll"], "val_ppl": val["ppl"],
               "val_bleu_greedy": bleu, "val_chrf_greedy": chrf, "epoch_seconds": epoch_s,
               "total_seconds": total_s, "mean_grad_norm": gn_sum / max(gn_n, 1), "skipped_steps": skipped}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writerow(row)
        print(f"epoch {epoch:>2} | train(dropout) {train_loss_dropout:.4f} | train(eval) {tr_eval['nll']:.4f} | "
              f"val {val['nll']:.4f} (ppl {val['ppl']:.1f}) | BLEU {bleu:.2f} chrF {chrf:.2f} | {epoch_s:.0f}s", flush=True)

        improved = val["nll"] < best_val - 1e-4
        if improved:
            best_val, bad_epochs, best_epoch = val["nll"], 0, epoch
            save_checkpoint(run_dir / "best.pt", model, cfg, src_tok.vocab_size, tgt_tok.vocab_size,
                            {"epoch": epoch, "val_loss": val["nll"], "train_loss_eval": tr_eval["nll"],
                             "val_bleu_greedy": bleu})
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_val": best_val, "bad_epochs": bad_epochs, "elapsed": total_s},
                   run_dir / "last.pt")
        if bad_epochs >= args.patience:
            stopped_early = True
            print(f"early stopping: no val improvement for {args.patience} epochs")
            break

    rows = list(csv.DictReader(open(log_path)))
    best_row = min(rows, key=lambda r: float(r["val_loss"])) if rows else {}
    summary = {"run_name": args.run_name, "cell": args.cell, "params": params, "device": str(device),
               "best_epoch": int(best_row["epoch"]) if rows else None,
               "best_val_loss": float(best_row["val_loss"]) if rows else None,
               "best_val_ppl": float(best_row["val_ppl"]) if rows else None,
               "train_loss_eval_at_best": float(best_row["train_loss_eval"]) if rows else None,
               "train_loss_dropout_at_best": float(best_row["train_loss_dropout"]) if rows else None,
               "val_bleu_greedy_at_best": float(best_row["val_bleu_greedy"]) if rows else None,
               "epochs_run": len(rows), "stopped_early": stopped_early,
               "total_train_seconds": float(rows[-1]["total_seconds"]) if rows else None,
               "mean_epoch_seconds": sum(float(r["epoch_seconds"]) for r in rows) / max(len(rows), 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
