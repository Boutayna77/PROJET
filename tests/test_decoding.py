from types import SimpleNamespace

import pytest
import torch

from darija_nmt.decoding import _banned_ngram_tokens, beam_search, greedy_decode
from darija_nmt.model import ModelConfig, Seq2Seq

TOK = SimpleNamespace(pad_id=0, unk_id=1, bos_id=2, eos_id=3)


def model(cell="lstm"):
    torch.manual_seed(1)
    return Seq2Seq(ModelConfig(cell=cell, emb_dim=16, hidden_size=24, proj_size=12, layers=2, dropout=0.0), 50, 60, 0).eval()


def test_banned_ngrams():
    assert _banned_ngram_tokens([2, 5, 6, 5, 6], 3) == [5]
    assert _banned_ngram_tokens([2, 5], 3) == []
    assert sorted(_banned_ngram_tokens([2, 5, 6], 1)) == [2, 5, 6]


@pytest.mark.parametrize("cell", ["lstm", "cifg", "lstmp"])
def test_beam1_equals_greedy_and_no_banned_tokens(cell):
    m = model(cell)
    for src_ids in ([5, 6, 7, 3], [11, 12, 13, 14, 15, 3]):
        g = greedy_decode(m, torch.tensor([src_ids]), 2, 3, [0, 1, 2], 12)[0]
        b = beam_search(m, src_ids, TOK, torch.device("cpu"), beam_size=1, length_penalty=1.0, no_repeat_ngram=0, max_len=12)
        assert g == b
        assert not ({0, 1, 2} & set(b))


def test_beam_runs_and_respects_no_repeat():
    m = model()
    out = beam_search(m, [5, 6, 7, 3], TOK, torch.device("cpu"), beam_size=4, no_repeat_ngram=1, max_len=15)
    assert len(set(out)) == len(out) and all(isinstance(t, int) for t in out)


def test_batched_greedy_matches_single():
    m = model()
    a, b = [5, 6, 7, 3], [11, 12, 13, 14, 15, 3]
    batch = torch.tensor([a + [0, 0], b])
    together = greedy_decode(m, batch, 2, 3, [0, 1, 2], 12)
    alone = [greedy_decode(m, torch.tensor([x]), 2, 3, [0, 1, 2], 12)[0] for x in (a, b)]
    assert together == alone


def test_block_immediate_repeat():
    m = model()
    out = beam_search(m, [5, 6, 7, 3], TOK, torch.device("cpu"), beam_size=3, no_repeat_ngram=0, max_len=20,
                      block_immediate_repeat=True)
    assert all(a != b for a, b in zip(out, out[1:]))
