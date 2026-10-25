"""Hand-written LSTM cells from the assignment appendix (no nn.LSTM / nn.LSTMCell anywhere).

Basic linear operation: nn.Linear. For speed the four (or three) per-gate input matrices
W_x* are stacked into ONE nn.Linear (`x2h`, with the bias b_*) and the per-gate recurrent
matrices W_h* into ONE bias-free nn.Linear (`h2h`). Stacking is mathematically identical to
the per-gate equations (unit tests compare against explicit per-gate references). Because the
input projection does not depend on the recurrence, sequence-level code can apply `x2h` to the
whole sequence at once and only run `step` (recurrent part) inside the time loop.

State convention: (h, c) with h in R^{output_size}, c in R^{hidden_size}.
  - Vanilla / CIFG / WMC : output_size == hidden_size
  - LSTMP                : output_size == proj_size (P < H)
"""
from __future__ import annotations

import torch
from torch import nn

State = tuple[torch.Tensor, torch.Tensor]


class _BaseCell(nn.Module):
    NAME = "base"
    N_GATES = 4
    FORGET_IDX = 1  # position of the forget gate inside the stacked gate vector
    USES_PROJ = False

    def __init__(self, input_size: int, hidden_size: int, proj_size: int | None = None,
                 init: str = "orthogonal", forget_bias: float = 1.0):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.output_size = proj_size if (self.USES_PROJ and proj_size) else hidden_size
        self.x2h = nn.Linear(input_size, self.N_GATES * hidden_size, bias=True)
        self.h2h = nn.Linear(self.output_size, self.N_GATES * hidden_size, bias=False)
        self._build_extra(init)
        self.reset_parameters(init, forget_bias)

    # hooks -------------------------------------------------------------------------------
    def _build_extra(self, init: str) -> None:
        pass

    def _extra_linears(self) -> list[nn.Linear]:
        return []

    # init --------------------------------------------------------------------------------
    @staticmethod
    def _init_weight(w: torch.Tensor, mode: str, recurrent: bool, n_blocks: int) -> None:
        with torch.no_grad():
            if mode == "uniform":
                nn.init.uniform_(w, -0.08, 0.08)
            elif mode == "orthogonal" and recurrent:
                for blk in w.chunk(n_blocks, dim=0):  # one orthogonal block per gate
                    nn.init.orthogonal_(blk)
            else:  # "xavier", or the input-side matrices in "orthogonal" mode
                nn.init.xavier_uniform_(w)

    def reset_parameters(self, init: str = "orthogonal", forget_bias: float = 1.0) -> None:
        if init not in {"orthogonal", "xavier", "uniform"}:
            raise ValueError(f"unknown init '{init}'")
        self._init_weight(self.x2h.weight, init, False, self.N_GATES)
        self._init_weight(self.h2h.weight, init, True, self.N_GATES)
        for lin in self._extra_linears():
            self._init_weight(lin.weight, init, True, 1)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        with torch.no_grad():
            self.x2h.bias.zero_()
            H, k = self.hidden_size, self.FORGET_IDX
            self.x2h.bias[k * H:(k + 1) * H] = forget_bias

    # API ---------------------------------------------------------------------------------
    def init_state(self, batch: int, ref: torch.Tensor) -> State:
        return ref.new_zeros(batch, self.output_size), ref.new_zeros(batch, self.hidden_size)

    def step(self, xg: torch.Tensor, state: State) -> State:  # xg = x2h(x_t)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, state: State) -> State:
        return self.step(self.x2h(x), state)


class VanillaLSTMCell(_BaseCell):
    """Eqs. (1)-(6).  Gate order in the stacked vectors: [i, f, o, g]."""
    NAME = "lstm"

    def step(self, xg, state):
        h, c = state
        i, f, o, g = (xg + self.h2h(h)).chunk(4, dim=-1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        return o * torch.tanh(c), c


class CIFGCell(_BaseCell):
    """Eqs. (13)-(18): coupled input-forget gate, i_t = 1 - f_t.  Order: [f, o, g] (3 gates)."""
    NAME = "cifg"
    N_GATES = 3
    FORGET_IDX = 0

    def step(self, xg, state):
        h, c = state
        f, o, g = (xg + self.h2h(h)).chunk(3, dim=-1)
        f, o, g = torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + (1.0 - f) * g
        return o * torch.tanh(c), c


class LSTMPCell(_BaseCell):
    """Eqs. (19)-(25): LSTM with a projection layer, h_t = W_p m_t + b_p, h_t in R^P (P < H)."""
    NAME = "lstmp"
    USES_PROJ = True

    def __init__(self, input_size, hidden_size, proj_size=None, **kw):
        if not proj_size:
            raise ValueError("LSTMP requires proj_size")
        super().__init__(input_size, hidden_size, proj_size, **kw)

    def _build_extra(self, init):
        self.proj = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def _extra_linears(self):
        return [self.proj]

    def step(self, xg, state):
        h, c = state
        i, f, o, g = (xg + self.h2h(h)).chunk(4, dim=-1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        m = o * torch.tanh(c)
        return self.proj(m), c


class WMCCell(_BaseCell):
    """Eqs. (7)-(12): working-memory connections. Gates see tanh(W_c* c). Order: [i, f, o, g].

    W_ci, W_cf, W_co are full H x H matrices (nn.Linear, no bias), as in the paper's
    'working memory connections'; the output gate reads the NEW cell state c_t (eq. 9).
    """
    NAME = "wmc"

    def _build_extra(self, init):
        H = self.hidden_size
        self.Wci = nn.Linear(H, H, bias=False)
        self.Wcf = nn.Linear(H, H, bias=False)
        self.Wco = nn.Linear(H, H, bias=False)

    def _extra_linears(self):
        return [self.Wci, self.Wcf, self.Wco]

    def step(self, xg, state):
        h, c_prev = state
        gi, gf, go, gg = (xg + self.h2h(h)).chunk(4, dim=-1)
        i = torch.sigmoid(gi + torch.tanh(self.Wci(c_prev)))
        f = torch.sigmoid(gf + torch.tanh(self.Wcf(c_prev)))
        g = torch.tanh(gg)
        c = f * c_prev + i * g
        o = torch.sigmoid(go + torch.tanh(self.Wco(c)))
        return o * torch.tanh(c), c


CELLS = {"lstm": VanillaLSTMCell, "wmc": WMCCell, "cifg": CIFGCell, "lstmp": LSTMPCell}


def build_cell(name: str, input_size: int, hidden_size: int, proj_size: int | None = None,
               init: str = "orthogonal", forget_bias: float = 1.0) -> _BaseCell:
    if name not in CELLS:
        raise ValueError(f"unknown cell '{name}', choose from {sorted(CELLS)}")
    return CELLS[name](input_size, hidden_size, proj_size=proj_size, init=init, forget_bias=forget_bias)
