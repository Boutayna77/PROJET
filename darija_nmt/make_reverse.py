"""Build the reverse (Darija -> English) task from the existing splits by swapping columns and tokenizers.
No model code changes are needed: train it with  --data-dir data/processed_rev --tok-dir data/tokenizer_rev.
The `group` column becomes the Darija key so multi-reference evaluation groups by Darija source.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .text_utils import arabic_key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tok-dir", default="data/tokenizer")
    ap.add_argument("--out-data", default="data/processed_rev")
    ap.add_argument("--out-tok", default="data/tokenizer_rev")
    args = ap.parse_args(argv)

    out = Path(args.out_data)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(Path(args.data_dir) / f"{split}.csv", dtype=str, keep_default_na=False)
        rev = pd.DataFrame({"id": df.id, "src": df.tgt, "tgt": df.src, "group": df.tgt.map(arabic_key)})
        rev.to_csv(out / f"{split}.csv", index=False, encoding="utf-8")
    tok = Path(args.out_tok)
    tok.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(args.tok_dir) / "tgt.model", tok / "src.model")   # Darija tokenizer becomes the source side
    shutil.copy(Path(args.tok_dir) / "src.model", tok / "tgt.model")   # English tokenizer becomes the target side
    print(f"reverse task written to {out} and {tok}")


if __name__ == "__main__":
    main()
