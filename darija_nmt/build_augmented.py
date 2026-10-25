"""Mix real training pairs with (tagged) synthetic back-translated pairs. val/test are copied UNCHANGED.

  python -m darija_nmt.build_augmented --synthetic data/synthetic/bt_filtered.csv --out-dir data/processed_bt
Train on the result with  --data-dir data/processed_bt  (same tokenizers). Evaluation stays on the real val/test.
Tagging: synthetic sources get the prefix "BT " so the model can tell noisy synthetic English from real English
(tagged back-translation); the tag is never used at test time.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

TAG = "BT "


def build(real: pd.DataFrame, syn: pd.DataFrame, max_synthetic: int | None = None, real_upsample: int = 1,
          tag: bool = True, seed: int = 0) -> pd.DataFrame:
    syn = syn.copy()
    if max_synthetic and len(syn) > max_synthetic:
        syn = syn.sort_values("chrf", ascending=False).head(max_synthetic) if "chrf" in syn.columns \
            else syn.sample(n=max_synthetic, random_state=seed)
    syn = syn.reset_index(drop=True)
    ids = [f"bt-{i}" for i in range(len(syn))]
    syn_df = pd.DataFrame({"id": ids, "src": (TAG + syn.src) if tag else syn.src, "tgt": syn.tgt, "group": ids})
    mixed = pd.concat([real] * max(real_upsample, 1) + [syn_df], ignore_index=True)
    return mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--synthetic", default="data/synthetic/bt_filtered.csv")
    ap.add_argument("--out-dir", default="data/processed_bt")
    ap.add_argument("--max-synthetic", type=int, default=None)
    ap.add_argument("--real-upsample", type=int, default=1, help="repeat the real pairs this many times")
    ap.add_argument("--no-tag", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    src_dir, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(src_dir / "train.csv", dtype=str, keep_default_na=False)
    syn = pd.read_csv(args.synthetic, dtype={"src": str, "tgt": str}, keep_default_na=False)
    mixed = build(real, syn, args.max_synthetic, args.real_upsample, not args.no_tag, args.seed)
    mixed.to_csv(out / "train.csv", index=False, encoding="utf-8")
    for split in ("val", "test"):
        shutil.copy(src_dir / f"{split}.csv", out / f"{split}.csv")
    n_syn = int(mixed.id.str.startswith("bt-").sum())
    rep = {"real_pairs": len(real), "real_upsample": args.real_upsample, "synthetic_pairs": n_syn,
           "total_train_pairs": len(mixed), "synthetic_share": n_syn / max(len(mixed), 1), "tagged": not args.no_tag}
    (out / "build_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
