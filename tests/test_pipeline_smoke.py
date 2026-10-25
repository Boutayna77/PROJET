"""End-to-end CLI smoke test on SYNTHETIC data (software check only; says nothing about Darija quality)."""
import csv
import json
import subprocess
import sys

import pytest

from tests.synthetic import make_corpus

pytest.importorskip("sentencepiece")
pytest.importorskip("sacrebleu")


def run(*args):
    r = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    assert r.returncode == 0, f"{args}\nSTDOUT:{r.stdout[-2000:]}\nSTDERR:{r.stderr[-3000:]}"
    return r.stdout


def test_end_to_end(tmp_path):
    raw, proc, tok, runs = tmp_path / "s.csv", tmp_path / "proc", tmp_path / "tok", tmp_path / "runs"
    make_corpus(500, seed=3).to_csv(raw, index=False, encoding="utf-8")
    run("darija_nmt.prepare_data", "--input", str(raw), "--out-dir", str(proc))
    run("darija_nmt.train_tokenizer", "--data-dir", str(proc), "--out-dir", str(tok), "--src-vocab", "80", "--tgt-vocab", "80")
    small = ["--data-dir", str(proc), "--tok-dir", str(tok), "--run-root", str(runs), "--hidden-size", "48", "--emb-dim", "32",
             "--proj-size", "24", "--layers", "2", "--batch-size", "32", "--device", "cpu", "--bleu-sentences", "20"]
    for cell in ("lstm", "cifg", "lstmp", "wmc"):
        run("darija_nmt.train", "--run-name", cell, "--cell", cell, "--epochs", "3", *small)
        rows = list(csv.DictReader(open(runs / cell / "log.csv")))
        assert len(rows) == 3 and float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"]), cell
        run("darija_nmt.evaluate", "--run-dir", str(runs / cell), "--split", "val", "--data-dir", str(proc),
            "--limit", "10", "--device", "cpu")
        res = json.loads((runs / cell / "eval_val_beam.json").read_text(encoding="utf-8"))
        assert 0 <= res["bleu"] <= 100 and res["n_sources"] == 10
    out = run("darija_nmt.translate", "--run-dirs", str(runs / "lstm"), "--text", "I see the house", "--device", "cpu")
    assert "[lstm" in out
    # resume path
    run("darija_nmt.train", "--run-name", "lstm", "--cell", "lstm", "--epochs", "4", "--resume", *small)
    assert len(list(csv.DictReader(open(runs / "lstm" / "log.csv")))) == 4
    run("darija_nmt.make_report_assets", "--runs", "lstm", "cifg", "lstmp", "wmc", "--run-root", str(runs), "--out-dir", str(tmp_path / "res"))
    assert (tmp_path / "res" / "results_table.md").exists()


def test_backtranslation_pipeline(tmp_path):
    """reverse model -> back-translate -> filter -> augmented training with warm start (synthetic data, software check)."""
    raw, proc, tok, runs = tmp_path / "s.csv", tmp_path / "proc", tmp_path / "tok", tmp_path / "runs"
    make_corpus(500, seed=3).to_csv(raw, index=False, encoding="utf-8")
    run("darija_nmt.prepare_data", "--input", str(raw), "--out-dir", str(proc))
    run("darija_nmt.train_tokenizer", "--data-dir", str(proc), "--out-dir", str(tok), "--src-vocab", "80", "--tgt-vocab", "80")
    mono = make_corpus(200, seed=99, noisy=False).darija_ar.str.rstrip(".")
    (proc / "mono_darija_ar.txt").write_text("\n".join(mono), encoding="utf-8")
    small = ["--hidden-size", "48", "--emb-dim", "32", "--proj-size", "24", "--layers", "2", "--batch-size", "32",
             "--device", "cpu", "--bleu-sentences", "20", "--run-root", str(runs)]
    run("darija_nmt.train", "--run-name", "lstm", "--cell", "lstm", "--epochs", "2", "--data-dir", str(proc), "--tok-dir", str(tok), *small)
    run("darija_nmt.make_reverse", "--data-dir", str(proc), "--tok-dir", str(tok), "--out-data", str(tmp_path / "rev"), "--out-tok", str(tmp_path / "revtok"))
    run("darija_nmt.train", "--run-name", "rev", "--cell", "lstm", "--epochs", "2", "--data-dir", str(tmp_path / "rev"), "--tok-dir", str(tmp_path / "revtok"), *small)
    syn = tmp_path / "syn"
    run("darija_nmt.backtranslate", "--reverse-run", str(runs / "rev"), "--mono", str(proc / "mono_darija_ar.txt"),
        "--out", str(syn / "raw.csv"), "--limit", "80", "--device", "cpu")
    run("darija_nmt.filter_synthetic", "--input", str(syn / "raw.csv"), "--data-dir", str(proc), "--forward-run", str(runs / "lstm"),
        "--scored-out", str(syn / "scored.csv"), "--out", str(syn / "filtered.csv"), "--keep-frac", "0.5", "--min-chrf", "0", "--device", "cpu")
    run("darija_nmt.filter_synthetic", "--scored-in", str(syn / "scored.csv"), "--out", str(syn / "filtered2.csv"), "--keep-frac", "0.25")
    run("darija_nmt.build_augmented", "--data-dir", str(proc), "--synthetic", str(syn / "filtered.csv"), "--out-dir", str(tmp_path / "bt"))
    run("darija_nmt.train", "--run-name", "lstm_bt", "--cell", "lstm", "--epochs", "2", "--data-dir", str(tmp_path / "bt"), "--tok-dir", str(tok), *small)
    run("darija_nmt.train", "--run-name", "lstm_bt_ft", "--cell", "lstm", "--epochs", "1", "--data-dir", str(proc), "--tok-dir", str(tok),
        "--init-from", str(runs / "lstm_bt" / "best.pt"), "--lr", "3e-4", *small)
    out = run("darija_nmt.evaluate", "--run-dir", str(runs / "lstm_bt_ft"), "--split", "val", "--data-dir", str(proc), "--limit", "10",
              "--decode", "beam", "--no-immediate-repeat", "--bootstrap", "5", "--device", "cpu")
    assert (runs / "lstm_bt_ft" / "eval_val_beam.json").exists()
