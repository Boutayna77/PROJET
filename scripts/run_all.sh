#!/usr/bin/env bash
# Full pipeline. Run from the repository root with the virtualenv active:  bash scripts/run_all.sh
# Every stage can also be run on its own (see README.md). Edit the variables below, not the stages.
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1          # let unsupported ops fall back to CPU instead of crashing
export TOKENIZERS_PARALLELISM=false

DEVICE="${DEVICE:-auto}"                      # auto = mps > cuda > cpu ; use DEVICE=cpu to compare speed
EPOCHS="${EPOCHS:-40}"
HP_PAIRS="${HP_PAIRS:-0}"                       # 0 = full (small) training split
HP_EPOCHS="${HP_EPOCHS:-10}"
# Final shared hyper-parameters: set these from results/hparam_search.csv AFTER the study
# (selection by validation loss). Defaults are the pre-study baseline.
LR="${LR:-1e-3}"; OPT="${OPT:-adam}"; BS="${BS:-128}"; INIT="${INIT:-orthogonal}"; DROPOUT="${DROPOUT:-0.3}"

echo "== 1. data: inspect, clean, split ==";  python -m darija_nmt.prepare_data --input data/raw/sentences.csv --out-dir data/processed
echo "== 2. tokenizers (train split only) =="; python -m darija_nmt.train_tokenizer --data-dir data/processed --out-dir data/tokenizer
echo "== 2b. transliteration audit ==";       python -m darija_nmt.audit_transliteration
echo "== 3. unit tests ==";                    python -m pytest -q
echo "== 4. speed benchmark (30 steps) ==";    python -m darija_nmt.train --run-name bench --run-root runs_bench --cell lstm --epochs 1 --max-steps-per-epoch 30 --bleu-every 0 --device "$DEVICE"

if [ "${SKIP_HPARAM:-0}" != "1" ]; then
  echo "== 5. hyper-parameter study (vanilla LSTM, 10 epochs each, no early stopping) =="
  python -m darija_nmt.hparam_search --cells lstm --pairs "$HP_PAIRS" --epochs "$HP_EPOCHS" --extra --device "$DEVICE"
fi

echo "== 6. final training: identical data, splits, hyper-parameters, seed for the three required models =="
for CELL in lstm cifg lstmp; do
  python -m darija_nmt.train --run-name "$CELL" --cell "$CELL" --epochs "$EPOCHS" --lr "$LR" --optimizer "$OPT" \
      --batch-size "$BS" --init "$INIT" --dropout "$DROPOUT" --device "$DEVICE" --resume
done

echo "== 7. choose decoding settings on VALIDATION =="
for CELL in lstm cifg lstmp; do
  python -m darija_nmt.evaluate --run-dir "runs/$CELL" --split val --decode beam --beam-size 5 --limit 500 --device "$DEVICE"
done

echo "== 8. final TEST evaluation (once per model) =="
for CELL in lstm cifg lstmp; do
  python -m darija_nmt.evaluate --run-dir "runs/$CELL" --split test --decode both --beam-size 5 --device "$DEVICE"
done

echo "== 9. tables and plots =="
python -m darija_nmt.make_report_assets --runs lstm cifg lstmp
echo "Done. See results/results_table.md, results/*.png, runs/*/examples_test_beam.md"
