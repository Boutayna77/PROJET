"""Greedy (batched) and beam-search (single sentence) decoding with repetition control."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .text_utils import normalize_english

NEG_INF = float("-inf")
MAX_SRC_TOKENS = 127


def encode_source(src_tok, text: str) -> list[int]:
    return src_tok.encode(normalize_english(text))[:MAX_SRC_TOKENS] + [src_tok.eos_id]


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, bos: int, eos: int, banned: list[int], max_len: int):
    """src (B,S) -> list of token-id lists (EOS stripped)."""
    model.eval()
    enc_out, mask, states, keys = model.encode(src)
    B, dev = src.size(0), src.device
    tok = torch.full((B,), bos, dtype=torch.long, device=dev)
    out = torch.full((B, max_len), model.pad_id, dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(max_len):
        logits, states = model.decoder.step(tok, states, enc_out, keys, mask)
        logits[:, banned] = NEG_INF
        tok = logits.argmax(dim=-1).masked_fill(finished, model.pad_id)
        out[:, t] = tok
        finished = finished | tok.eq(eos)
        if bool(finished.all()):
            break
    res = []
    for row in out.tolist():
        ids = []
        for i in row:
            if i == eos or i == model.pad_id:
                break
            ids.append(i)
        res.append(ids)
    return res


@torch.no_grad()
def greedy_translate(model, src_tok, tgt_tok, texts, device, batch_size: int = 64, max_len_cap: int = 160):
    model.eval()
    ids = [encode_source(src_tok, t) for t in texts]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    banned = [tgt_tok.pad_id, tgt_tok.bos_id, tgt_tok.unk_id]
    out = [""] * len(ids)
    for s in range(0, len(order), batch_size):
        bidx = order[s:s + batch_size]
        S = max(len(ids[i]) for i in bidx)
        src = torch.full((len(bidx), S), src_tok.pad_id, dtype=torch.long)
        for r, i in enumerate(bidx):
            src[r, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
        hyp = greedy_decode(model, src.to(device), tgt_tok.bos_id, tgt_tok.eos_id, banned,
                            min(max_len_cap, int(S * 1.6) + 10))
        for r, i in enumerate(bidx):
            out[i] = tgt_tok.decode(hyp[r])
    return out


def _banned_ngram_tokens(seq: list[int], n: int) -> list[int]:
    """Tokens that would create an n-gram already present in seq (seq includes BOS)."""
    if n <= 0 or len(seq) < n:
        return []
    prefix = tuple(seq[len(seq) - (n - 1):]) if n > 1 else ()
    return [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == prefix]


@torch.no_grad()
def beam_search(model, src_ids: list[int], tgt_tok, device, beam_size: int = 5,
                length_penalty: float = 0.8, no_repeat_ngram: int = 3, max_len: int | None = None,
                block_immediate_repeat: bool = False) -> list[int]:
    """Beam search for ONE source sentence; the beam is the batch dimension.

    Hypotheses are ranked by sum(log p) / len**length_penalty. A hypothesis ending in EOS is
    moved to the finished list; search stops when `beam_size` hypotheses finished (or max_len).
    """
    model.eval()
    k, bos, eos = beam_size, tgt_tok.bos_id, tgt_tok.eos_id
    banned = [tgt_tok.pad_id, tgt_tok.bos_id, tgt_tok.unk_id]
    max_len = max_len or min(160, int(len(src_ids) * 1.6) + 10)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    enc_out, mask, states, keys = model.encode(src)
    enc_out, mask = enc_out.repeat(k, 1, 1), mask.repeat(k, 1)
    keys = keys.repeat(k, 1, 1) if keys is not None else None
    states = [(h.repeat(k, 1), c.repeat(k, 1)) for h, c in states]

    seqs = [[bos] for _ in range(k)]
    scores = torch.full((k,), NEG_INF, device=device)
    scores[0] = 0.0  # only one live hypothesis at the start
    finished: list[tuple[float, list[int]]] = []

    for _ in range(max_len):
        tok = torch.tensor([s[-1] for s in seqs], dtype=torch.long, device=device)
        logits, states = model.decoder.step(tok, states, enc_out, keys, mask)
        logp = F.log_softmax(logits.float(), dim=-1)
        logp[:, banned] = NEG_INF
        if no_repeat_ngram > 0:
            for b, s in enumerate(seqs):
                bt = _banned_ngram_tokens(s, no_repeat_ngram)
                if bt:
                    logp[b, bt] = NEG_INF
        if block_immediate_repeat:  # never emit the same token twice in a row (e.g. "ديالي ديالي")
            for b, s in enumerate(seqs):
                if len(s) > 1:
                    logp[b, s[-1]] = NEG_INF
        V = logp.size(1)
        cand = (scores.unsqueeze(1) + logp).view(-1)
        top_s, top_i = cand.topk(min(2 * k, cand.numel()))
        new_seqs, new_scores, src_beam = [], [], []
        for rank, (sc, idx) in enumerate(zip(top_s.tolist(), top_i.tolist())):
            if sc == NEG_INF:
                break
            b, w = divmod(idx, V)
            if w == eos:
                if rank < k:
                    finished.append((sc / (len(seqs[b]) ** length_penalty), seqs[b][1:]))
            elif len(new_seqs) < k:
                new_seqs.append(seqs[b] + [w])
                new_scores.append(sc)
                src_beam.append(b)
        if not new_seqs or len(finished) >= k:
            break
        while len(new_seqs) < k:  # pad with dead hypotheses so the batch size stays k
            new_seqs.append(new_seqs[0])
            new_scores.append(NEG_INF)
            src_beam.append(src_beam[0])
        order = torch.tensor(src_beam, dtype=torch.long, device=device)
        states = [(h.index_select(0, order), c.index_select(0, order)) for h, c in states]
        seqs, scores = new_seqs, torch.tensor(new_scores, dtype=torch.float, device=device)

    if not finished:  # ran out of length: fall back to the live hypotheses
        for s, sc in zip(seqs, scores.tolist()):
            if sc != NEG_INF:
                finished.append((sc / (max(len(s) - 1, 1) ** length_penalty), s[1:]))
    if not finished:
        return []
    return max(finished, key=lambda x: x[0])[1]


def beam_translate(model, src_tok, tgt_tok, text: str, device, **kw) -> str:
    return tgt_tok.decode(beam_search(model, encode_source(src_tok, text), tgt_tok, device, **kw))
