"""Shared helpers: seeding, device, loss, loss evaluation, optimiser factory."""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seq_loss(logits: torch.Tensor, target: torch.Tensor, pad_id: int, label_smoothing: float = 0.0):
    """Returns (loss to back-propagate, summed UNSMOOTHED NLL, number of target tokens).

    Label smoothing follows the PyTorch definition: (1-eps)*NLL + eps*mean_k(-log p_k).
    All reported losses (train/val/test) are the plain NLL in nats per target token (EOS included,
    BOS and padding excluded) so that smoothing never inflates the reported numbers.
    """
    lp = F.log_softmax(logits.float(), dim=-1)
    nll = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = target.ne(pad_id)
    n = mask.sum().clamp(min=1)
    nll_sum = (nll * mask).sum()
    if label_smoothing > 0:
        smooth = -lp.mean(dim=-1)
        loss = (((1 - label_smoothing) * nll + label_smoothing * smooth) * mask).sum() / n
    else:
        loss = nll_sum / n
    return loss, nll_sum.detach(), n


@torch.no_grad()
def evaluate_loss(model, loader, device, pad_id: int) -> dict:
    model.eval()
    total, count = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        _, nll_sum, n = seq_loss(logits, tgt[:, 1:], pad_id, 0.0)
        total += float(nll_sum.item())
        count += int(n.item())
    nll = total / max(count, 1)
    return {"nll": nll, "ppl": math.exp(min(nll, 50.0)), "tokens": count, "nll_sum": total}


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.9):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay if weight_decay > 0 else 0.01)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer '{name}'")
