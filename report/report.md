# English → Moroccan Darija Neural Machine Translation with LSTM Variants
*Advanced NLP – RNN lab project, 2026/27. Team: ⟨names⟩ · Code: ⟨public GitHub URL⟩*

> **Template status:** methodology below describes what the code does. Every ⟨FILL⟩ must be replaced with numbers produced by the
> pipeline (`data/processed/data_report.json`, `results/results_table.md`, `runs/*/examples_test_beam.md`). Do not add numbers
> that did not come from a run. Target length: 4–5 pages — compress sections 3–4 when pasting results.

## 1. Task and goals
Translate English into Moroccan Darija **written in Arabic script** (`darija_ar`) with three LSTM-based models that share data
splits, tokenisation, architecture and protocol: a Vanilla-LSTM baseline and two non-vanilla cells (CIFG, LSTMP). Goals: preserve meaning,
produce natural Arabic-script Darija, avoid Latin leakage and repetition loops, and compare cells at equal protocol and reported parameter counts.

## 2. Exploratory data analysis
Source: `sentences.csv` (eng, darija, darija_ar), **87,785 rows**; numbers below are from `results/data_audit/data_report.json` and `transliteration_audit.json`.
- **Supervision is scarce:** only **12,743 rows (14.5%)** have an English sentence; 75,042 rows (85.5%) are Darija-only. No missing values in `darija`/`darija_ar`.
- Duplicates: 75 exact rows, 1,199 repeated English strings, 336 repeated (English, Arabic) pairs. Script of `darija_ar`: 87,476 Arabic, 309 mixed with Latin letters; English: 100% Latin.
- Length (words, median / p95): English 5 / 13, Darija 4 / 8; 11.7% of English entries have 1–2 words (dictionary-like entries and interjections).
- Cleaning funnel (12,743 → **12,208 pairs**): swapped English/Darija columns −4 (3 truly swapped, 1 false positive), Latin letters in target −5, length-ratio outliers −4, numbers corrupted in the target −12, digit mismatch −1, duplicate pairs after normalisation −509. Split by English source (90/5/5): train 11,003 / val 604 / test 601 pairs (10,034 / 557 / 557 unique English sentences), zero source overlap (asserted); 3.6% / 2.3% of val / test target strings also occur in train.
- Vocabulary: train Darija has 48,879 tokens and 14,218 word types (68% singletons); 19.9% of val/test Darija tokens are unseen in train (English 6.3%). Hence subword units are required.
- **Quality of `darija_ar`:** both Darija columns have identical word counts in every row, and the Latin word alone determines the Arabic spelling for **99.6%** of 371,277 aligned tokens (94.8% of frequent types have one single rendering). `darija_ar` is therefore a rule-based transliteration of chat-style Latin Darija, with artifacts such as `حايات سيحييا` (`7ayat si7iya`), `صاحيح` (`Sa7i7`), `أُ` for “and”, and chat digits turned into letters inside numbers (`10:30`→`10:ع0`). Models trained on it imitate this convention; BLEU/chrF measure agreement with it, not naturalness.
- Manual audit of 100 random cleaned pairs (`audit_sample.csv`) by a Darija speaker: ⟨FILL aligned %, natural %, typical problems⟩. (My own non-native reading of 25 pairs found no misalignment.)
Alignment heuristics (length-ratio MAD, digit sets, Latin/Arabic word counts) detect gross misalignment only; they cannot certify translation correctness.

## 3. Modelling approach
**Preprocessing.** Light Arabic normalisation (diacritics, tatweel, Persian letters, digits, Arabic punctuation, elongation); alef unification off.
**Tokenisation.** SentencePiece unigram, 3k English / 4k Darija vocabulary; English side trained on the training split only, Darija side also on 74,335 unlabeled Darija-only sentences (val/test targets excluded). Tokens/word: ⟨FILL⟩, unk rate on val: ⟨FILL⟩.
**Architecture.** 2-layer bidirectional encoder, 2-layer decoder, Luong general attention, tied target embeddings, dropout 0.3; hidden H=256 (LSTMP P=128), embedding 256.
**Cells** (hand-written with `nn.Linear`, checked in unit tests against explicit per-gate transcriptions of the equations and, for the vanilla cell, `nn.LSTMCell` — ⟨confirm the tests passed⟩):
Vanilla LSTM (eqs. 1–6); CIFG: i_t = 1 − f_t, three gates (eqs. 13–18); LSTMP: m_t = o_t⊙tanh(c_t), h_t = W_p m_t + b_p with P=256<H (eqs. 19–25).
Parameters (total / recurrent): Vanilla ⟨FILL⟩ / ⟨FILL⟩, CIFG ⟨FILL⟩ / ⟨FILL⟩, LSTMP ⟨FILL⟩ / ⟨FILL⟩.
The two additional cells were chosen *before* any result: CIFG tests whether tying the gates loses accuracy while saving parameters; LSTMP tests the parameter/accuracy trade-off of a recurrent bottleneck. ⟨Optional: WMC ablation⟩.
**Training/decoding.** Adam, clip 1.0, label smoothing 0.1 (reported losses are unsmoothed NLL), ReduceLROnPlateau, early stopping, checkpoint = best validation loss; beam 5, length penalty 0.8, no-repeat-trigram.

## 4. Hyper-parameter study
One-factor-at-a-time (full training split, 10 epochs per run, vanilla LSTM): learning rate, optimizer, batch size, initialisation, dropout, hidden size (figure `results/hparam_lstm.png`).
Findings: ⟨FILL: effect sizes and the chosen values, with validation-loss evidence⟩. Limitation: short runs and single seed; interactions not explored; tuning on the baseline may favour it.

## 5. Results
Same splits and protocol for all models; checkpoints chosen on validation; test used once.

⟨PASTE results/results_table.md, trimmed: params, train loss (eval mode), val loss, test loss/ppl, test BLEU & chrF (beam), greedy BLEU, pure-Arabic rate, repeated-trigram rate, training time⟩

Learning curves: `results/training_curves.png`.
**Loss goals.** Train loss < 2: ⟨achieved? value⟩. Validation loss < 2: ⟨achieved? value⟩. Interpretation: ⟨FILL — mention tokenisation dependence, label noise, multiple valid translations; report nats/char⟩.
**Qualitative analysis** (from `examples_test_beam.md`): ⟨2–3 successful and 3–4 failed examples per model with EN / REF / HYP, and diagnosis: lexical choice, dialect vs MSA, spelling variants, hallucination, repetition, Latin leakage⟩.
**Cell comparison.** ⟨FILL: differences vs seed-to-seed variance (run ≥2 extra seeds if time permits, otherwise state single-seed caveat), parameter efficiency, training time/epoch, decoding time/sentence⟩.

## 6. Limitations and improvements
Automatic metrics under-reward valid alternative Darija spellings/wordings; references may be noisy, machine-generated or in a different register; the target orthography is machine-generated (see §2), so naturalness cannot be established from this corpus; Latin-script code-switching is filtered out by design; only ~600 test pairs (report bootstrap intervals); single seed unless stated; the 74k Darija-only sentences are used only for the vocabulary.
Improvements (to be justified by the error analysis): ⟨decoder language-model pre-training or back-translation on the 74k Darija-only sentences, subword regularisation, spelling-variant canonicalisation, coverage/length-aware attention, larger or ensembled models, human evaluation by Darija speakers⟩.
