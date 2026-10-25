#!/usr/bin/env bash
# Back-translation experiment. Run AFTER the baseline runs (runs/lstm runs/cifg runs/lstmp) exist.
# Use the SAME hyper-parameters as the baselines (edit the variables, do not change them per model).
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
DEV="${DEV:-mps}"; H="${H:-256}"; LR="${LR:-1e-3}"; OPT="${OPT:-adam}"; BS="${BS:-128}"; INIT="${INIT:-orthogonal}"; DROP="${DROP:-0.3}"
KEEP="${KEEP:-0.5}"      # fraction of synthetic pairs kept by round-trip chrF
COMMON="--hidden-size $H --proj-size $((H/2)) --lr $LR --optimizer $OPT --batch-size $BS --init $INIT --dropout $DROP --device $DEV"

echo "== 1. reverse task (Darija -> English) =="
python -m darija_nmt.make_reverse
python -m darija_nmt.train --run-name rev_lstm --cell lstm --data-dir data/processed_rev --tok-dir data/tokenizer_rev --epochs 40 $COMMON --resume
python -m darija_nmt.evaluate --run-dir runs/rev_lstm --split val --data-dir data/processed_rev --decode greedy --bootstrap 0 --device "$DEV"

echo "== 2. back-translate the 74k unlabeled Darija sentences =="
python -m darija_nmt.backtranslate --reverse-run runs/rev_lstm --device "$DEV"

echo "== 3. filter (heuristics + round-trip with the real-data baseline) =="
python -m darija_nmt.filter_synthetic --forward-run runs/lstm --keep-frac "$KEEP" --device "$DEV"

echo "== 4. mix real + synthetic (val/test untouched) =="
python -m darija_nmt.build_augmented --synthetic data/synthetic/bt_filtered.csv --out-dir data/processed_bt

echo "== 5. train all three cells on the augmented data, then fine-tune on real data only =="
for CELL in lstm cifg lstmp; do
  python -m darija_nmt.train --run-name "${CELL}_bt" --cell "$CELL" --data-dir data/processed_bt --epochs 40 $COMMON --resume
  python -m darija_nmt.train --run-name "${CELL}_bt_ft" --cell "$CELL" --data-dir data/processed --init-from "runs/${CELL}_bt/best.pt" \
      --epochs 15 --hidden-size "$H" --proj-size $((H/2)) --lr 3e-4 --optimizer "$OPT" --batch-size "$BS" --init "$INIT" --dropout "$DROP" --device "$DEV" --resume
done

echo "== 6. compare on VALIDATION (same decoding as the baselines) =="
for R in lstm lstm_bt lstm_bt_ft cifg cifg_bt cifg_bt_ft lstmp lstmp_bt lstmp_bt_ft; do
  python -m darija_nmt.evaluate --run-dir "runs/$R" --split val --decode beam --beam-size 5 --limit 300 --bootstrap 0 --device "$DEV" --tag bt_compare
done
echo "Pick the variant per validation chrF/BLEU, then run the TEST evaluation once for the chosen runs."
