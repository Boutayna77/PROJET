"""Encoder-decoder NMT model built from the custom LSTM cells.

Encoder : embedding -> L stacked bidirectional layers (each: forward cell + backward cell,
          outputs concatenated and fed to the next layer). Padding is handled by masked state
          updates (state is frozen on pad positions), so the backward direction starts at each
          sentence's real last token without packing sequences.
Bridge  : tanh(nn.Linear([fwd; bwd])) maps the encoder's final per-layer states to the decoder's
          initial (h, c).
Decoder : embedding -> L stacked unidirectional layers (teacher forcing runs layer-by-layer over
          the whole target; inference runs step-by-step - both give identical logits, unit-tested).
Attention: Luong "general" global attention (no input feeding => the recurrent part can be run
          for all target positions first and attention applied in one batched matmul).
          out = W_o tanh(W_c [context; h]); with --tie-weights W_o shares the target embedding.
The same encoder-decoder, attention and output layer are used for every cell type, so only the
recurrent cell differs between the compared models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .cells import build_cell


@dataclass
class ModelConfig:
    cell: str = "lstm"          # lstm | cifg | lstmp | wmc
    emb_dim: int = 256
    hidden_size: int = 256      # H (cell-state size)
    proj_size: int = 128        # P, only used by lstmp
    layers: int = 2
    dropout: float = 0.3
    attention: bool = True
    tie_weights: bool = True
    init: str = "orthogonal"    # orthogonal | xavier | uniform
    forget_bias: float = 1.0

    def to_dict(self):
        return asdict(self)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.fwd, self.bwd = nn.ModuleList(), nn.ModuleList()
        in_size = cfg.emb_dim
        for _ in range(cfg.layers):
            f = build_cell(cfg.cell, in_size, cfg.hidden_size, cfg.proj_size, cfg.init, cfg.forget_bias)
            b = build_cell(cfg.cell, in_size, cfg.hidden_size, cfg.proj_size, cfg.init, cfg.forget_bias)
            self.fwd.append(f)
            self.bwd.append(b)
            in_size = 2 * f.output_size
        self.out_size = in_size  # 2P

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        """src (B,S) long, mask (B,S) bool -> outputs (B,S,2P), finals per layer."""
        B, S = src.shape
        x = self.drop(self.emb(src))
        mf = mask.unsqueeze(-1).to(x.dtype)  # (B,S,1)
        finals = []
        n_layers = len(self.fwd)
        for l in range(n_layers):
            fc, bc = self.fwd[l], self.bwd[l]
            xf, xb = fc.x2h(x), bc.x2h(x)  # (B,S,G*H) input projections for the whole sequence
            hf, cf = fc.init_state(B, x)
            hb, cb = bc.init_state(B, x)
            out_f, out_b = [], [None] * S
            for t in range(S):
                m = mask[:, t].unsqueeze(1)
                nh, nc = fc.step(xf[:, t], (hf, cf))
                hf, cf = torch.where(m, nh, hf), torch.where(m, nc, cf)
                out_f.append(hf * mf[:, t])
                tb = S - 1 - t
                mb = mask[:, tb].unsqueeze(1)
                nh, nc = bc.step(xb[:, tb], (hb, cb))
                hb, cb = torch.where(mb, nh, hb), torch.where(mb, nc, cb)
                out_b[tb] = hb * mf[:, tb]
            x = torch.cat([torch.stack(out_f, 1), torch.stack(out_b, 1)], dim=-1)
            finals.append((hf, hb, cf, cb))
            if l < n_layers - 1:
                x = self.drop(x)
        return x, finals


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int, enc_out_size: int):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.cells = nn.ModuleList()
        in_size = cfg.emb_dim
        for _ in range(cfg.layers):
            c = build_cell(cfg.cell, in_size, cfg.hidden_size, cfg.proj_size, cfg.init, cfg.forget_bias)
            self.cells.append(c)
            in_size = c.output_size
        P = in_size
        self.use_attention = cfg.attention
        if cfg.attention:
            self.key_proj = nn.Linear(enc_out_size, P, bias=False)
            self.comb = nn.Linear(enc_out_size + P, cfg.emb_dim)
        else:
            self.comb = nn.Linear(P, cfg.emb_dim)
        self.out = nn.Linear(cfg.emb_dim, vocab_size)
        if cfg.tie_weights:
            self.out.weight = self.emb.weight

    def keys(self, enc_out):
        return self.key_proj(enc_out) if self.use_attention else None

    def output_layer(self, h, enc_out, keys, mask):
        """h (B,T,P) -> logits (B,T,V)."""
        if self.use_attention:
            scores = torch.bmm(h, keys.transpose(1, 2))                 # (B,T,S)
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            ctx = torch.bmm(torch.softmax(scores, dim=-1), enc_out)     # (B,T,2P)
            feat = torch.cat([ctx, h], dim=-1)
        else:
            feat = h
        return self.out(self.drop(torch.tanh(self.comb(feat))))

    def forward_seq(self, tgt_in, states, enc_out, keys, mask):
        """Teacher-forced pass. tgt_in (B,T) -> logits (B,T,V)."""
        x = self.drop(self.emb(tgt_in))
        T = x.size(1)
        n_layers = len(self.cells)
        for l, cell in enumerate(self.cells):
            xg = cell.x2h(x)
            h, c = states[l]
            outs = []
            for t in range(T):
                h, c = cell.step(xg[:, t], (h, c))
                outs.append(h)
            x = torch.stack(outs, dim=1)
            if l < n_layers - 1:
                x = self.drop(x)
        return self.output_layer(x, enc_out, keys, mask)

    def step(self, tok, states, enc_out, keys, mask):
        """One inference step. tok (B,) -> logits (B,V), new states."""
        x = self.drop(self.emb(tok))
        new_states = []
        n_layers = len(self.cells)
        for l, cell in enumerate(self.cells):
            h, c = cell(x, states[l])
            new_states.append((h, c))
            x = self.drop(h) if l < n_layers - 1 else h
        logits = self.output_layer(x.unsqueeze(1), enc_out, keys, mask).squeeze(1)
        return logits, new_states


class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig, src_vocab: int, tgt_vocab: int, pad_id: int = 0):
        super().__init__()
        self.cfg, self.pad_id = cfg, pad_id
        self.encoder = Encoder(src_vocab, cfg, pad_id)
        self.decoder = Decoder(tgt_vocab, cfg, pad_id, self.encoder.out_size)
        P = self.decoder.cells[0].output_size
        H = cfg.hidden_size
        self.bridge_h = nn.ModuleList([nn.Linear(2 * P, P) for _ in range(cfg.layers)])
        self.bridge_c = nn.ModuleList([nn.Linear(2 * H, H) for _ in range(cfg.layers)])
        for emb in (self.encoder.emb, self.decoder.emb):
            nn.init.normal_(emb.weight, mean=0.0, std=cfg.emb_dim ** -0.5)
            with torch.no_grad():
                emb.weight[pad_id].zero_()

    def encode(self, src: torch.Tensor):
        mask = src.ne(self.pad_id)
        enc_out, finals = self.encoder(src, mask)
        states = []
        for l, (hf, hb, cf, cb) in enumerate(finals):
            h = torch.tanh(self.bridge_h[l](torch.cat([hf, hb], dim=-1)))
            c = torch.tanh(self.bridge_c[l](torch.cat([cf, cb], dim=-1)))
            states.append((h, c))
        return enc_out, mask, states, self.decoder.keys(enc_out)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        enc_out, mask, states, keys = self.encode(src)
        return self.decoder.forward_seq(tgt_in, states, enc_out, keys, mask)

    # reporting -----------------------------------------------------------------------------
    def param_breakdown(self) -> dict:
        n = lambda mods: sum(p.numel() for m in mods for p in m.parameters() if p.requires_grad)  # noqa: E731
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)  # shared tensors counted once
        enc_cells = n([self.encoder.fwd, self.encoder.bwd])
        dec_cells = n([self.decoder.cells])
        return {"total": total, "encoder_cells": enc_cells, "decoder_cells": dec_cells,
                "recurrent_cells_total": enc_cells + dec_cells}
