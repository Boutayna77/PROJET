"""One-factor-at-a-time hyper-parameter study on a reduced budget (validation loss decides).

Each run is a short training on a random subset (see --pairs/--epochs). Results go to
results/hparam_search.csv; completed runs are skipped when you re-launch.
  python -m darija_nmt.hparam_search --cells lstm --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

BASE = {"lr": 1e-3, "optimizer": "adam", "batch_size": 128, "init": "orthogonal", "dropout": 0.3, "hidden_size": 256}
OPT_LR = {"adam": 1e-3, "adamw": 1e-3, "rmsprop": 3e-4, "sgd": 0.2}
GRID = {
    "lr": [3e-4, 1e-3, 3e-3],
    "optimizer": ["adam", "adamw", "rmsprop", "sgd"],
    "batch_size": [32, 64, 128, 256],
    "init": ["orthogonal", "xavier", "uniform"],
    "dropout": [0.1, 0.3, 0.5],
    "hidden_size": [128, 256, 512],
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=["lstm"])
    ap.add_argument("--factors", nargs="+", default=list(GRID), choices=list(GRID))
    ap.add_argument("--pairs", type=int, default=0, help="0 = full training split (it is only ~11k pairs)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--run-root", default="runs_hparam")
    ap.add_argument("--out", default="results/hparam_search.csv")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to darija_nmt.train")
    args = ap.parse_args(argv)

    rows = []
    for cell in args.cells:
        for factor in args.factors:
            for value in GRID[factor]:
                cfg = dict(BASE)
                cfg[factor] = value
                if factor == "optimizer":
                    cfg["lr"] = OPT_LR[value]
                name = f"hp_{cell}_{factor}_{value}"
                summ = Path(args.run_root) / name / "summary.json"
                if not summ.exists():
                    cmd = [sys.executable, "-m", "darija_nmt.train", "--run-name", name, "--run-root", args.run_root,
                           "--cell", cell, "--lr", str(cfg["lr"]), "--optimizer", cfg["optimizer"],
                           "--batch-size", str(cfg["batch_size"]), "--init", cfg["init"], "--dropout", str(cfg["dropout"]),
                           "--hidden-size", str(cfg["hidden_size"]), "--proj-size", str(cfg["hidden_size"] // 2),
                           "--epochs", str(args.epochs), "--patience", "99", "--bleu-every", "0",
                           *(["--max-train-pairs", str(args.pairs)] if args.pairs > 0 else []), *args.extra]
                    print("RUN", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=False)
                if summ.exists():
                    s = json.loads(summ.read_text())
                    rows.append({"cell": cell, "factor": factor, "value": value, "lr_used": cfg["lr"],
                                 "best_val_loss": s["best_val_loss"], "best_val_ppl": s["best_val_ppl"],
                                 "train_loss_eval": s["train_loss_eval_at_best"], "epochs": s["epochs_run"],
                                 "seconds": s["total_train_seconds"]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["cell"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
