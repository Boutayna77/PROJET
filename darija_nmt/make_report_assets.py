"""Collect run results into results/results_table.{md,csv} and plots (no numbers are typed by hand)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"lstm": "Vanilla LSTM", "cifg": "CIFG", "lstmp": "LSTMP", "wmc": "WMC"}


def read_log(path: Path):
    return list(csv.DictReader(open(path)))


def maybe(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["lstm", "cifg", "lstmp"])
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in args.runs:
        d = Path(args.run_root) / name
        summ = maybe(d / "summary.json")
        if not summ:
            print(f"skip {name}: no summary.json")
            continue
        label = LABELS.get(summ["cell"], summ["cell"]) + ("" if name == summ["cell"] else f" [{name}]")
        log = read_log(d / "log.csv")
        ep = [int(r["epoch"]) for r in log]
        axes[0].plot(ep, [float(r["train_loss_eval"]) for r in log], marker="o", label=f"{label} train")
        axes[0].plot(ep, [float(r["val_loss"]) for r in log], marker="s", linestyle="--", label=f"{label} val")
        axes[1].plot(ep, [float(r["val_bleu_greedy"]) for r in log], marker="o", label=label)
        tb, tg = maybe(d / "eval_test_beam.json"), maybe(d / "eval_test_greedy.json")
        row = {"model": label, "trainable_params": summ["params"]["total"],
               "recurrent_params": summ["params"]["recurrent_cells_total"], "best_epoch": summ["best_epoch"],
               "train_loss_eval": summ["train_loss_eval_at_best"], "val_loss": summ["best_val_loss"],
               "test_loss": (tb or tg or {}).get("loss", {}).get("nll"),
               "test_ppl": (tb or tg or {}).get("loss", {}).get("ppl"),
               "test_nats_per_char": (tb or tg or {}).get("loss", {}).get("nll_per_target_char"),
               "test_bleu_beam": tb and tb["bleu"], "test_chrf_beam": tb and tb["chrf"],
               "test_bleu_greedy": tg and tg["bleu"], "test_chrf_greedy": tg and tg["chrf"],
               "pure_arabic_rate_beam": tb and tb["pure_arabic_script_rate"],
               "repeated_trigram_rate_beam": tb and tb["repeated_trigram_rate"],
               "length_ratio_beam": tb and tb["length_ratio_hyp_over_ref"],
               "train_minutes": summ["total_train_seconds"] / 60, "sec_per_epoch": summ["mean_epoch_seconds"]}
        table.append(row)
    axes[0].set(xlabel="epoch", ylabel="NLL (nats / target token)", title="Loss (train measured in eval mode)")
    axes[1].set(xlabel="epoch", ylabel="greedy BLEU (500 val sentences)", title="Validation BLEU")
    for a in axes:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150)

    if table:
        keys = list(table[0])
        with open(out / "results_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(table)
        fmt = lambda v: "n/a" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
        md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in table]
        (out / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))

    hp = Path(out / "hparam_search.csv")
    if hp.exists():
        rows = list(csv.DictReader(open(hp)))
        factors = sorted({r["factor"] for r in rows})
        for cell in sorted({r["cell"] for r in rows}):
            fig, axs = plt.subplots(1, len(factors), figsize=(3.2 * len(factors), 3.4))
            axs = [axs] if len(factors) == 1 else axs
            for ax, fac in zip(axs, factors):
                sub = [r for r in rows if r["factor"] == fac and r["cell"] == cell]
                ax.bar([r["value"] for r in sub], [float(r["best_val_loss"]) for r in sub])
                ax.set_title(fac, fontsize=9); ax.set_ylabel("best val NLL"); ax.tick_params(axis="x", labelsize=7)
            fig.suptitle(f"One-factor-at-a-time search ({LABELS.get(cell, cell)}, reduced budget)", fontsize=10)
            fig.tight_layout(); fig.savefig(out / f"hparam_{cell}.png", dpi=150)


if __name__ == "__main__":
    main()
