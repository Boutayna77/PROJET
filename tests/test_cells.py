import pytest
import torch

from darija_nmt.cells import CELLS, build_cell

B, E, H, P = 5, 7, 6, 4
sig, tanh = torch.sigmoid, torch.tanh


def make(name, init="orthogonal", dtype=torch.float32):
    torch.manual_seed(0)
    cell = build_cell(name, E, H, proj_size=P, init=init).to(dtype)
    with torch.no_grad():  # non-trivial biases / weights so that every term of the equations matters
        for p in cell.parameters():
            p.normal_(0, 0.5)
    return cell


def rand_state(cell, dtype=torch.float32):
    return torch.randn(B, cell.output_size, dtype=dtype), torch.randn(B, cell.hidden_size, dtype=dtype)


def reference_step(name, cell, x, h, c):
    """Explicit, per-gate transcription of the appendix equations using the cell's own weights."""
    Wx, bx, Wh = cell.x2h.weight, cell.x2h.bias, cell.h2h.weight
    sl = lambda k: slice(k * H, (k + 1) * H)  # noqa: E731
    gx = lambda k: x @ Wx[sl(k)].T + bx[sl(k)]  # noqa: E731
    gh = lambda k: h @ Wh[sl(k)].T  # noqa: E731
    if name == "lstm":      # eq 1-6, order i f o g
        i, f, o, g = sig(gh(0) + gx(0)), sig(gh(1) + gx(1)), sig(gh(2) + gx(2)), tanh(gh(3) + gx(3))
        c2 = f * c + i * g
        return o * tanh(c2), c2
    if name == "cifg":      # eq 13-18, order f o g, i = 1 - f
        f, o, g = sig(gh(0) + gx(0)), sig(gh(1) + gx(1)), tanh(gh(2) + gx(2))
        c2 = f * c + (1 - f) * g
        return o * tanh(c2), c2
    if name == "lstmp":     # eq 19-25
        i, f, o, g = sig(gh(0) + gx(0)), sig(gh(1) + gx(1)), sig(gh(2) + gx(2)), tanh(gh(3) + gx(3))
        c2 = f * c + i * g
        m = o * tanh(c2)
        return m @ cell.proj.weight.T + cell.proj.bias, c2
    if name == "wmc":       # eq 7-12
        i = sig(gx(0) + gh(0) + tanh(c @ cell.Wci.weight.T))
        f = sig(gx(1) + gh(1) + tanh(c @ cell.Wcf.weight.T))
        g = tanh(gh(3) + gx(3))
        c2 = f * c + i * g
        o = sig(gx(2) + gh(2) + tanh(c2 @ cell.Wco.weight.T))
        return o * tanh(c2), c2
    raise AssertionError(name)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_matches_equations(name):
    cell = make(name)
    x = torch.randn(B, E)
    h, c = rand_state(cell)
    h1, c1 = cell(x, (h, c))
    hr, cr = reference_step(name, cell, x, h, c)
    assert h1.shape == (B, cell.output_size) and c1.shape == (B, H)
    assert torch.allclose(h1, hr, atol=1e-6) and torch.allclose(c1, cr, atol=1e-6)


def test_vanilla_matches_torch_lstmcell():
    cell = make("lstm")
    ref = torch.nn.LSTMCell(E, H)
    order = [0, 1, 3, 2]  # ours [i f o g] -> torch [i f g o]
    with torch.no_grad():
        ref.weight_ih.copy_(torch.cat([cell.x2h.weight[k * H:(k + 1) * H] for k in order]))
        ref.weight_hh.copy_(torch.cat([cell.h2h.weight[k * H:(k + 1) * H] for k in order]))
        ref.bias_ih.copy_(torch.cat([cell.x2h.bias[k * H:(k + 1) * H] for k in order]))
        ref.bias_hh.zero_()
    x = torch.randn(B, E)
    h, c = rand_state(cell)
    h1, c1 = cell(x, (h, c))
    h2, c2 = ref(x, (h, c))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_cifg_is_vanilla_with_tied_gates():
    """Vanilla with W_i = -W_f, b_i = -b_f gives i = sigmoid(-z_f) = 1 - f: must equal CIFG."""
    cg, va = make("cifg"), make("lstm")
    with torch.no_grad():
        sl = lambda k: slice(k * H, (k + 1) * H)  # noqa: E731
        va.x2h.weight[sl(0)] = -cg.x2h.weight[sl(0)]; va.x2h.bias[sl(0)] = -cg.x2h.bias[sl(0)]
        va.h2h.weight[sl(0)] = -cg.h2h.weight[sl(0)]
        va.x2h.weight[sl(1)] = cg.x2h.weight[sl(0)]; va.x2h.bias[sl(1)] = cg.x2h.bias[sl(0)]
        va.h2h.weight[sl(1)] = cg.h2h.weight[sl(0)]
        va.x2h.weight[sl(2)] = cg.x2h.weight[sl(1)]; va.x2h.bias[sl(2)] = cg.x2h.bias[sl(1)]
        va.h2h.weight[sl(2)] = cg.h2h.weight[sl(1)]
        va.x2h.weight[sl(3)] = cg.x2h.weight[sl(2)]; va.x2h.bias[sl(3)] = cg.x2h.bias[sl(2)]
        va.h2h.weight[sl(3)] = cg.h2h.weight[sl(2)]
    x = torch.randn(B, E)
    h, c = rand_state(cg)
    a, b = cg(x, (h, c)), va(x, (h, c))
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_backward_and_gradcheck(name):
    cell = make(name, dtype=torch.float64)
    x = torch.randn(B, E, dtype=torch.float64, requires_grad=True)
    h, c = (t.requires_grad_() for t in rand_state(cell, torch.float64))
    h1, c1 = cell(x, (h, c))
    (h1.sum() + c1.sum()).backward()
    for n, p in cell.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
    assert torch.autograd.gradcheck(lambda x_, h_, c_: cell(x_, (h_, c_)), (x.detach().requires_grad_(), h.detach().requires_grad_(), c.detach().requires_grad_()))


def test_parameter_counts_and_gate_structure():
    n = lambda c: sum(p.numel() for p in c.parameters())  # noqa: E731
    v, cg, lp, wm = (build_cell(k, 16, 32, proj_size=8) for k in ("lstm", "cifg", "lstmp", "wmc"))
    assert v.x2h.out_features == 4 * 32 and cg.x2h.out_features == 3 * 32
    assert n(v) == 4 * 32 * (16 + 32) + 4 * 32
    assert n(cg) == 3 * 32 * (16 + 32) + 3 * 32
    assert n(wm) == n(v) + 3 * 32 * 32
    assert lp.output_size == 8 and lp.h2h.in_features == 8 and n(lp) == 4 * 32 * (16 + 8) + 4 * 32 + 32 * 8 + 8
    assert n(cg) < n(v) < n(wm) and n(lp) < n(v)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_init_modes_and_forget_bias(name):
    for init in ("orthogonal", "xavier", "uniform"):
        cell = build_cell(name, 16, 32, proj_size=8, init=init)
        k = cell.FORGET_IDX
        assert torch.all(cell.x2h.bias[k * 32:(k + 1) * 32] == 1.0)
        assert torch.isfinite(cell.x2h.weight).all() and torch.isfinite(cell.h2h.weight).all()
    cell = build_cell("lstm", 16, 32, init="orthogonal")
    for blk in cell.h2h.weight.chunk(4, 0):
        assert torch.allclose(blk.T @ blk, torch.eye(32), atol=1e-4)
