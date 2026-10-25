#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
DEV="${DEV:-mps}"; H="${H:-256}"; LR="${LR:-1e-3}"; DROP="${DROP:-0.3}"; INIT="${INIT:-orthogonal}"; BS=32
CELLS="${CELLS:-lstm cifg lstmp}"
COMMON="--hidden-size $H --proj-size $((H/2)) --lr $LR --optimizer adam --batch-size $BS --init $INIT --dropout $DROP --device $DEV"

for c in $CELLS; do
  echo "== $c : real data, batch size $BS =="
  python -m darija_nmt.train --run-name ${c}_b32 --cell $c --epochs 40 $COMMON --resume
done
for c in $CELLS; do
  echo "== $c : back-translation data, batch size $BS =="
  python -m darija_nmt.train --run-name ${c}_bt_b32 --cell $c --data-dir data/processed_bt --epochs 40 $COMMON --resume
  python -m darija_nmt.train --run-name ${c}_bt_ft_b32 --cell $c --data-dir data/processed --init-from runs/${c}_bt_b32/best.pt --epochs 15 \
      --hidden-size $H --proj-size $((H/2)) --lr 3e-4 --optimizer adam --batch-size $BS --init $INIT --dropout $DROP --device $DEV --resume
done

echo "== validation loss of every run (lower is better) =="
python - << 'PYEOF'
import json, glob
for f in sorted(glob.glob('runs/*/summary.json')):
    s = json.load(open(f))
    print(f"{s['run_name']:18} val loss {s['best_val_loss']:.3f}   train(eval) {s['train_loss_eval_at_best']:.3f}   best epoch {s['best_epoch']}")
PYEOF
