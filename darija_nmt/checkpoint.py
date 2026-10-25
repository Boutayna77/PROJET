"""Self-contained run directories: config, tokenizers, best/last checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import ModelConfig, Seq2Seq
from .tokenizer import SPTokenizer


def save_checkpoint(path, model, cfg: ModelConfig, src_vocab: int, tgt_vocab: int, extra: dict) -> None:
    torch.save({"model": model.state_dict(), "model_cfg": cfg.to_dict(), "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab, **extra}, str(path))


def load_run(run_dir, device, which: str = "best.pt"):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / which, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Seq2Seq(cfg, ckpt["src_vocab"], ckpt["tgt_vocab"], pad_id=0)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    src_tok = SPTokenizer(run_dir / "tokenizer" / "src.model")
    tgt_tok = SPTokenizer(run_dir / "tokenizer" / "tgt.model")
    return model, src_tok, tgt_tok, ckpt
