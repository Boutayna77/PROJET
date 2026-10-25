import math

import pytest
import torch

from darija_nmt.checkpoint import save_checkpoint
from darija_nmt.model import ModelConfig, Seq2Seq
from darija_nmt.utils import seq_loss

SV, TV = 50, 60
CELL_NAMES = ["lstm", "cifg", "lstmp", "wmc"]


def tiny(cell="lstm", **kw):
    torch.manual_seed(0)
    cfg = ModelConfig(cell=cell, emb_dim=16, hidden_size=24, proj_size=12, layers=2, dropout=0.0, **kw)
    return Seq2Seq(cfg, SV, TV, 0)


def batch():
    src = torch.tensor([[5, 6, 7, 8, 9, 3, 0], [11, 12, 3, 0, 0, 0, 0], [20, 21, 22, 23, 3, 0, 0]])
    tgt = torch.tensor([[2, 5, 6, 7, 3, 0], [2, 9, 3, 0, 0, 0], [2, 4, 8, 9, 10, 3]])
    return src, tgt


@pytest.mark.parametrize("cell", CELL_NAMES)
def test_shapes_loss_and_backward(cell):
    m = tiny(cell)
    src, tgt = batch()
    logits = m(src, tgt[:, :-1])
    assert logits.shape == (3, 5, TV)
    loss, nll_sum, n = seq_loss(logits, tgt[:, 1:], 0, 0.1)
    assert int(n) == int((tgt[:, 1:] != 0).sum()) and torch.isfinite(loss)
    assert abs(float(nll_sum / n) - math.log(TV)) < 1.0     # roughly uniform at initialisation
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("cell", CELL_NAMES)
def test_teacher_forcing_equals_stepwise(cell):
    m = tiny(cell).eval()
    src, tgt = batch()
    tgt_in = tgt[:, :-1]
    full = m(src, tgt_in)
    enc_out, mask, states, keys = m.encode(src)
    for t in range(tgt_in.size(1)):
        lg, states = m.decoder.step(tgt_in[:, t], states, enc_out, keys, mask)
        assert torch.allclose(lg, full[:, t], atol=1e-5), (cell, t)


@pytest.mark.parametrize("cell", CELL_NAMES)
def test_padding_invariance(cell):
    """A sentence must get identical logits alone and inside a padded batch."""
    m = tiny(cell).eval()
    src, tgt = batch()
    logits = m(src, tgt[:, :-1])
    alone = m(src[1:2, :3], tgt[1:2, :3])
    assert torch.allclose(logits[1, :2], alone[0, :2], atol=1e-5)
    alone0 = m(src[0:1, :6], tgt[0:1, :-1][:, :4])
    assert torch.allclose(logits[0, :4], alone0[0, :4], atol=1e-5)


def test_variants_no_attention_untied():
    for kw in ({"attention": False}, {"tie_weights": False}, {"attention": False, "tie_weights": False}):
        m = tiny("lstm", **kw)
        src, tgt = batch()
        out = m(src, tgt[:, :-1])
        assert out.shape == (3, 5, TV)
        out.sum().backward()


def test_parameter_ordering_between_cells():
    p = {c: tiny(c).param_breakdown() for c in CELL_NAMES}
    assert p["cifg"]["total"] < p["lstm"]["total"] < p["wmc"]["total"]
    assert p["lstmp"]["total"] < p["lstm"]["total"]
    # shared tensors (tied embedding / output) are counted once
    m = tiny("lstm")
    assert p["lstm"]["total"] == sum(q.numel() for q in {id(q): q for q in m.parameters()}.values())


@pytest.mark.parametrize("cell", CELL_NAMES)
def test_can_overfit_one_batch(cell):
    m = tiny(cell)
    src, tgt = batch()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(120):
        opt.zero_grad()
        loss, _, _ = seq_loss(m(src, tgt[:, :-1]), tgt[:, 1:], 0, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.3 * first, (cell, first, float(loss))


def test_checkpoint_roundtrip(tmp_path):
    m = tiny("lstmp").eval()
    src, tgt = batch()
    ref = m(src, tgt[:, :-1])
    save_checkpoint(tmp_path / "c.pt", m, m.cfg, SV, TV, {"epoch": 1})
    ck = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=False)
    m2 = Seq2Seq(ModelConfig(**ck["model_cfg"]), ck["src_vocab"], ck["tgt_vocab"], 0)
    m2.load_state_dict(ck["model"])
    m2.eval()
    assert torch.allclose(ref, m2(src, tgt[:, :-1]), atol=1e-6)
