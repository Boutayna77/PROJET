"""Datasets, length-bucketed batching and evaluation groups (multi-reference)."""
from __future__ import annotations

import math
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class ParallelDataset(Dataset):
    """Pre-tokenised (src ids + EOS, BOS + tgt ids + EOS) pairs."""

    def __init__(self, csv_path=None, src_tok=None, tgt_tok=None, max_len: int = 128,
                 limit: int | None = None, seed: int = 0, _from=None):
        if _from is not None:  # internal: build from already-encoded lists
            self.src_ids, self.tgt_ids, self.src_text, self.tgt_text, self.n_truncated = _from
            return
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        self.src_text, self.tgt_text = df.src.tolist(), df.tgt.tolist()
        self.src_ids, self.tgt_ids, self.n_truncated = [], [], 0
        for s, t in zip(self.src_text, self.tgt_text):
            si, ti = src_tok.encode(s), tgt_tok.encode(t)
            if len(si) > max_len - 1 or len(ti) > max_len - 2:
                self.n_truncated += 1
            self.src_ids.append(si[:max_len - 1] + [src_tok.eos_id])
            self.tgt_ids.append([tgt_tok.bos_id] + ti[:max_len - 2] + [tgt_tok.eos_id])

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, i):
        return self.src_ids[i], self.tgt_ids[i]

    def take(self, n: int, seed: int = 0) -> "ParallelDataset":
        idx = random.Random(seed).sample(range(len(self)), min(n, len(self)))
        pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
        return ParallelDataset(_from=(pick(self.src_ids), pick(self.tgt_ids), pick(self.src_text),
                                      pick(self.tgt_text), 0))


class BucketBatchSampler(Sampler):
    """Shuffle, group into pools, sort inside each pool by length, cut batches, shuffle batches."""

    def __init__(self, lengths, batch_size: int, shuffle: bool, seed: int = 0, pool_mult: int = 50):
        self.lengths, self.bs, self.shuffle, self.seed, self.pool_mult = lengths, batch_size, shuffle, seed, pool_mult
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        pool = self.bs * self.pool_mult if self.shuffle else len(idx)
        batches = []
        for i in range(0, len(idx), max(pool, 1)):
            chunk = sorted(idx[i:i + pool], key=lambda j: self.lengths[j])
            batches += [chunk[k:k + self.bs] for k in range(0, len(chunk), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def make_collate(pad_id: int):
    def collate(batch):
        S = max(len(b[0]) for b in batch)
        T = max(len(b[1]) for b in batch)
        src = torch.full((len(batch), S), pad_id, dtype=torch.long)
        tgt = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for r, (s, t) in enumerate(batch):
            src[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            tgt[r, :len(t)] = torch.tensor(t, dtype=torch.long)
        return src, tgt
    return collate


def make_loader(ds: ParallelDataset, batch_size: int, shuffle: bool, pad_id: int = 0, seed: int = 0) -> DataLoader:
    lengths = [len(s) + len(t) for s, t in zip(ds.src_ids, ds.tgt_ids)]
    sampler = BucketBatchSampler(lengths, batch_size, shuffle, seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate(pad_id), num_workers=0)


def load_eval_groups(csv_path, limit: int | None = None, seed: int = 0):
    """One entry per unique English source with ALL its Darija references."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    groups: dict[str, tuple[str, list[str]]] = {}
    for s, t, g in zip(df.src, df.tgt, df.group):
        entry = groups.setdefault(g, (s, []))
        if t not in entry[1]:
            entry[1].append(t)
    srcs = [v[0] for v in groups.values()]
    refs = [v[1] for v in groups.values()]
    if limit is not None and limit < len(srcs):
        idx = sorted(random.Random(seed).sample(range(len(srcs)), limit))
        srcs, refs = [srcs[i] for i in idx], [refs[i] for i in idx]
    return srcs, refs
