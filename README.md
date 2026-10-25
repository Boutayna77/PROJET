# English → Moroccan Darija (Arabic script) NMT with hand-written LSTM cells

Advanced NLP lab project (RNN, 2026/27). Everything is implemented from scratch in PyTorch:
data audit, cleaning and leakage-safe splitting, tokenizers, **four custom LSTM cells written with
`nn.Linear` (no `nn.LSTM`/`nn.LSTMCell` in the model)**, an encoder–decoder with attention, training,
evaluation, decoding and an inference CLI.

| Role in the assignment | Cell (`--cell`) | Appendix equations |
|---|---|---|
| Baseline | `lstm` – Vanilla LSTM (stacking layers is still vanilla) | (1)–(6) |
| Additional model 1 | `cifg` – coupled input–forget gate | (13)–(18) |
| Additional model 2 | `lstmp` – LSTM with projection (P=256 < H=512) | (19)–(25) |
| Optional extra / ablation | `wmc` – working-memory connections | (7)–(12) |

## Verification status — read this first

This repository was written in a sandbox **without PyTorch, without network access and without the dataset**
(the CSV arrived later and was used only for the data audit). Therefore:

| Item | Status |
|---|---|
| Data pipeline (`prepare_data`, `audit_transliteration`) and text utilities | **Executed on the real `sentences.csv`** (outputs in `results/data_audit/`) and unit-tested on a synthetic noisy corpus. |
| All PyTorch code (cells, model, training, decoding, evaluation, CLI) | Written and byte-compiled, **never executed**. Run `python -m pytest -q` first; the tests were written to catch errors but their outcome is unknown. |
| Trained checkpoints, loss/BLEU numbers, plots, example translations | **Do not exist yet** (the dataset statistics do — see below). They are produced by `scripts/run_all.sh`. Nothing in this repo or the report template contains invented results. |

If a test fails on your machine, fix it before any long training run — the failing test names point at the culprit.

## What the real data showed (from `results/data_audit/`, produced by the code in this repo)

| Fact | Value |
|---|---|
| Raw rows | 87,785 |
| Rows with an English sentence (= supervised pairs) | **12,743 (14.5%)**; the other 75,042 rows (85.5%) are Darija-only |
| Missing `darija` / `darija_ar` | 0 / 0 |
| Exact duplicate rows · duplicate English strings · duplicate (eng, darija_ar) pairs | 75 · 1,199 · 336 |
| Script of `darija_ar` | 87,476 purely Arabic, 309 mixed with Latin letters (only 5 of those have an English sentence) |
| Words per sentence (median / p95) | English 5 / 13 · Darija 4 / 8 (very short sentences; 11.7% of English entries have 1–2 words) |
| Usable pairs after cleaning | **12,208** → train 11,003 · val 604 · test 601 (10,034 / 557 / 557 unique English sources; zero English overlap between splits) |
| Train word tokens / types (Darija) | 48,879 / 14,218; 68% of Darija types occur once; **19.9%** of val/test Darija word tokens are unseen in train (English: 6.3%) → word-level output is infeasible, subwords are required |
| Unlabeled Darija sentences kept for the target vocabulary | 74,335 |

**Most important finding — `darija_ar` is a mechanical transliteration of the Latin column, not independent Arabic-script Darija.**
Both columns have identical word counts in all 87,785 rows, and for the 371,277 aligned tokens the Latin word alone fixes the
Arabic spelling in **99.6%** of cases (94.8% of frequent word types have exactly one Arabic rendering; `audit_transliteration.py`).
Typical artifacts: `7ayat si7iya`→`حايات سيحييا`, `Sa7i7`→`صاحيح`, `ra2i3`→`راإع`, `liya`→`لييا`, `o` (“and”)→`أُ`, chat digits
turned into letters inside real numbers (`10:30`→`10:ع0`, 12 such rows removed), unconverted French letters (`é`, `ġ`; 5 rows removed).
Consequences: (1) a model trained here learns to emit *this transliteration convention*, so output can be faithful to the references and still
not look like how a Moroccan would type Arabic-script Darija; (2) BLEU/chrF against these references measures conformity to the converter,
not naturalness; (3) real naturalness can only be judged by native speakers (use `audit_sample.csv` and the generated example files).
Without an external, independently written Arabic-script Darija resource I cannot fix this inside the project; it is reported as a limitation.
Other defects found and handled: 3 rows with the English/Darija columns swapped (the filter removed 4 rows: 3 swapped + 1 legitimate loanword row), 509 duplicate pairs after normalisation.
A manual read of 25 random pairs found plausible alignment in all of them, but that is a non-native reading and not a substitute for a Darija-speaker audit.

**Practical consequences for the experiments.** With ~11k short pairs the models are heavily over-parameterised, so the defaults are
small (H=256, P=128, tokenizers 3k English / 4k Darija, dropout 0.3) and hidden size is part of the hyper-parameter study. An epoch is ~86 batches,
so training should be quick; the hyper-parameter search runs on the full training split. Expect strong overfitting, modest BLEU and a validation
loss that may stay above 2; with only ~600 test pairs, differences of a point or two of BLEU between cells are within noise, which is why
`evaluate.py` reports 95% bootstrap intervals (and extra seeds are recommended).

## Improving translation quality: back-translation (uses the 74k Darija-only sentences)

Why: 11k pairs of short sentences are too few for the models to learn how to *combine* phrases, so long or clause-heavy
inputs fail. The corpus also holds 74,335 human-written Darija sentences with no English. Back-translation turns them into
extra training pairs: a Darija→English model writes *synthetic English* for each Darija sentence; the English→Darija model then trains on
(synthetic English → real Darija). The target side stays real Darija, which is what the decoder needs to learn fluent output.
Everything stays LSTM-based. Steps (see `scripts/run_backtranslation.sh`):

1. `make_reverse` builds the Darija→English task by swapping columns/tokenizers (no model code changes); train it as `rev_lstm`.
2. `backtranslate` greedily translates all unlabeled Darija sentences.
3. `filter_synthetic` removes bad pairs: empty/Arabic-containing English, length-ratio outliers, repeated words, English equal to a val/test source,
   >3 variants per English; then a **round-trip filter**: the baseline English→Darija model (trained on real data only) re-translates the synthetic
   English and only the pairs whose output is closest (sentence chrF) to the original Darija are kept (`--keep-frac`).
4. `build_augmented` mixes real + synthetic pairs, prefixing synthetic sources with the tag `BT ` (tagged back-translation, so the model can tell
   noisy synthetic English from real English). val/test files are copied unchanged, so every comparison uses the same real val/test sets.
5. Train all three cells on the augmented data (`<cell>_bt`), optionally fine-tune on the real pairs only (`<cell>_bt_ft`, `--init-from`, lower LR).
6. Choose among baseline / `_bt` / `_bt_ft` and the `--keep-frac` value (e.g. 1.0, 0.5, 0.25 via `filter_synthetic --scored-in`) **on validation**, then run the test set once.

Honest caveats: the reverse model is itself trained on only 11k pairs, so synthetic English is noisy and the effect can be small or even
negative; the experiment is only worthwhile if validation chrF/BLEU improve. The Darija side of synthetic pairs may be near-duplicates of val/test
sentences with different spelling (only exact matches are excluded). Nothing here has been run yet. Also available now: `--no-immediate-repeat`
(evaluate/translate) to stop loops like "ديالي ديالي", and `analyze_errors` for quality by sentence length.

## Setup (macOS, Apple Silicon)

```bash
cd darija-nmt
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'MPS available:', torch.backends.mps.is_available())"
mkdir -p data/raw && cp /path/to/sentences.csv data/raw/sentences.csv
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Run step by step

```bash
# 1. Inspect, clean, split. Read data/processed/data_report.json and dropped_samples.csv.
python -m darija_nmt.prepare_data --input data/raw/sentences.csv --out-dir data/processed
python -m darija_nmt.audit_transliteration

# 2. Subword tokenizers (training split only)
python -m darija_nmt.train_tokenizer --data-dir data/processed --out-dir data/tokenizer

# 3. Unit + end-to-end smoke tests (synthetic data). MUST pass before training.
python -m pytest -q

# 4. Speed benchmark: seconds/step decides epochs & model size. Compare --device mps vs cpu.
python -m darija_nmt.train --run-name bench --run-root runs_bench --cell lstm --epochs 1 --max-steps-per-epoch 30 --bleu-every 0 --device mps
python -m darija_nmt.train --run-name bench --run-root runs_bench2 --cell lstm --epochs 1 --max-steps-per-epoch 30 --bleu-every 0 --device cpu

# 5. Hyper-parameter study (vanilla LSTM, full small training split, 10 epochs per run, one factor at a time)
python -m darija_nmt.hparam_search --cells lstm --epochs 10

# 6. Final training – identical data/splits/hyper-parameters/seed for the 3 required models (resumable)
for c in lstm cifg lstmp; do python -m darija_nmt.train --run-name $c --cell $c --epochs 40 --resume; done
#    optional ablations: --cell wmc ; --no-attention ; extra seeds via --seed 43 --run-name lstm_s43 ; ...

# 7. Pick decoding settings on VALIDATION, then evaluate on TEST once per model
python -m darija_nmt.evaluate --run-dir runs/lstm --split val --decode beam --beam-size 5 --limit 500
python -m darija_nmt.evaluate --run-dir runs/lstm --split test --decode both --beam-size 5

# 8. Tables and plots → results/
python -m darija_nmt.make_report_assets --runs lstm cifg lstmp

# 9. Translate
python -m darija_nmt.translate --run-dirs runs/lstm --text "How are you?"
python -m darija_nmt.translate --run-dirs runs/lstm runs/cifg runs/lstmp      # interactive, side by side
```
`bash scripts/run_all.sh` chains everything (after the hyper-parameter study, set `LR/OPT/BS/INIT/DROPOUT`
to the validation-selected values). Beam-search evaluation is single-sentence; on 3–4k test sentences it
takes minutes per model (`--device cpu` can be faster than MPS for these tiny batches).
macOS Terminal prints Arabic reversed and unjoined (a display issue, not a model issue): run `pip install arabic-reshaper python-bidi` (translate then reshapes Arabic for Terminal by default; `--raw` disables it, `--out file.txt` saves the real text). `predictions_*.csv` and `examples_*.md` open correctly in any editor.

## Key design decisions (and why)

1. **Target = `darija_ar` only.** The Latin `darija` column is never a training target; it is used only as an
   independent *alignment check* (its word count must agree with the Arabic column).
2. **Cleaning is explicit and measured** (`data_report.json` funnel): missing/blank fields, swapped columns (chat-digit words in the English column), numbers corrupted by the transliteration, wrong script
   (English containing Arabic letters; target without Arabic letters or with *any* Latin letter by default,
   `--max-latin-ratio`), URLs, length outliers, robust length-ratio outliers (median/MAD), digit sets that are
   disjoint across sides, Latin-vs-Arabic Darija word-count mismatch, normalised-duplicate pairs.
   Normalisation of the Arabic side is deliberately light: NFKC, diacritics/tatweel/invisible marks removed,
   Persian look-alike letters mapped, digits → ASCII, `? , ;` → `؟ ، ؛`, elongation 3+→2. Alef/hamza unification
   is **off** by default (`--normalize-alef`) because it changes legitimate spelling.
   A 100-pair `audit_sample.csv` is generated for manual alignment/naturalness checking — the heuristics cannot
   prove translations are *correct*, only catch gross misalignment.
3. **Leakage-safe split.** Whole English-key groups (case/punctuation-insensitive) go to one split (90/5/5), so
   paraphrase-variants of one source never straddle splits. The report records how often a test *target* string
   also occurs in train (a legitimate overlap, but a diagnostic for memorisation).
   Val/test BLEU/chrF use **all** Darija variants of a source as multiple references (Darija spelling is not unique).
4. **Tokenisation: SentencePiece unigram, separate vocabularies (3k English, 4k Darija), sized for ~49k Darija training tokens.** English side trained on the training split only; the Darija side additionally uses the 74k Darija-only sentences (unlabeled, val/test targets excluded). Try 2k/4k/8k and compare by chrF/BLEU, not loss.
   Word-level would give many OOVs for a morphologically rich, orthographically unstable dialect; character-level
   would make sequences 4–5× longer, which is costly for a Python-loop LSTM on a laptop. Digits are split.
   Vocabulary size changes the *per-token* loss mechanically; compare losses only at equal tokenisation, and use
   the tokenisation-independent nats/char, BLEU and chrF across settings.
5. **Architecture (shared by all cells):** 2-layer bidirectional encoder, 2-layer decoder (H=256, emb 256, LSTMP P=128), Luong "general"
   attention without input-feeding (so the decoder recurrence is computed for all target positions first), tied
   target embedding/output, dropout 0.3. Only the recurrent cell differs between compared models.
   Padding is handled by masked state updates, so no sequence packing is needed (covered by unit tests – not yet run: a sentence should get
   identical logits alone and inside a padded batch; teacher-forced and step-wise decoding should agree).
6. **Cell choice, fixed before any result was seen:** CIFG (does coupling the gates cost accuracy while saving ~25%
   of gate parameters?) and LSTMP (does a smaller recurrent projection P<H trade parameters for accuracy?).
   Both are cheaper than the baseline, which suits a laptop. WMC is implemented, has unit tests (not yet run) and is available as an extra run.
   Parameter counts (total and recurrent) are reported for every model.
7. **Optimisation:** Adam 1e-3, grad-clip 1.0, label smoothing 0.1, ReduceLROnPlateau (×0.5, patience 1), early stopping
   (patience 4), forget-gate bias 1, orthogonal recurrent init. These are *starting points*; `hparam_search` varies
   learning rate, optimizer, batch size, initialisation, dropout and hidden size, and selection is by validation loss.
   Caveat: tuning on the vanilla model only may favour the baseline; for a stricter study run
   `--cells lstm cifg lstmp` for at least `lr` and `init`.
8. **Losses are always plain NLL (nats/target token, EOS included, no label smoothing).** The log reports training loss
   both with dropout active (running average) and in eval mode on 4000 training pairs; the latter is the comparable
   number against validation loss.
9. **Decoding:** beam 5, length penalty 0.8, no-repeat-trigram (subword level) to prevent loops; tune these on val.
   Greedy is also reported. Diagnostics: pure-Arabic-script rate, any-Latin rate, repeated-trigram rate, length ratio,
   exact-match rate.
10. **Metrics:** corpus BLEU (text pre-tokenised so Arabic punctuation is split; sacreBLEU `tokenize=none`) and chrF
    (character-level, more robust to Darija's spelling variation), plus loss/perplexity. BLEU is a weak proxy for Darija:
    spellings vary, several translations are equally valid, and references may themselves be noisy or machine-generated.
    Read `examples_test_beam.md` (best, worst and random samples) before trusting any number.

## About the "loss < 2" goal

It is treated as a hypothesis, not a target to engineer. Per-token loss depends on vocabulary size (a smaller vocabulary
lowers it mechanically), on label noise and on the number of valid translations per source, which set a floor on validation
loss that no architecture can beat. Train loss < 2 is plausible; validation loss < 2 is uncertain and will be reported honestly
either way. The tokenizer will not be changed merely to hit the number.

## Layout
```
darija_nmt/   cells.py model.py decoding.py metrics.py dataset.py tokenizer.py train.py evaluate.py translate.py
              prepare_data.py train_tokenizer.py hparam_search.py make_report_assets.py text_utils.py utils.py checkpoint.py
tests/        cell equations vs explicit references / nn.LSTMCell, model invariants, decoding, data pipeline, CLI smoke test
scripts/run_all.sh     report/report.md (EDA filled from the real data; results to fill)     results/data_audit/ (real audit outputs)
runs/<name>/  config.json log.csv best.pt last.pt summary.json tokenizer/ eval_*.json predictions_*.csv examples_*.md
```
