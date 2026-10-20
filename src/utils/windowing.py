# -*- coding: utf-8 -*-
"""Fixed-window training and sliding-window inference (Chapter 2, A0_WINDOW=<frames>).

Instead of padding every trial to the longest one (48 % padding on the wind-turbine set), training samples are
random crops of W frames from each trial, drawn anew every epoch (ceil(n / W) crops per trial, so an epoch sees
about as many frames as before). Validation uses fixed tiles of W frames. At test time each trial is covered by
windows with hop W/2 and every frame takes its prediction from the window that gives it the most context:
the most central one (non-causal), or the one with the most past frames that still contains the frame's
look-ahead (causal, LOOKAHEAD=L). Trials of at most W frames are run whole.
Splits are (inputs, metadata, targets, masks) lists of per-trial tensors padded to the same length.
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def _n_valid(mask):
    return int(mask.sum().item())


class RandomWindowDataset(Dataset):
    """Random W-frame crops; ceil(n / W) items per trial per epoch."""

    def __init__(self, split, window):
        self.x, self.m, self.y, self.k = split
        self.W = int(window)
        self.items = [i for i, mk in enumerate(self.k) for _ in range(max(1, math.ceil(_n_valid(mk) / self.W)))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i = self.items[j]
        n = _n_valid(self.k[i])
        s = 0 if n <= self.W else int(torch.randint(0, n - self.W + 1, (1,)))
        return self.x[i][s:s + self.W], self.m[i], self.y[i][s:s + self.W], self.k[i][s:s + self.W]


class TiledWindowDataset(Dataset):
    """Deterministic W-frame tiles covering each trial (last tile aligned to the trial end)."""

    def __init__(self, split, window):
        self.x, self.m, self.y, self.k = split
        self.W = int(window)
        self.items = []
        for i, mk in enumerate(self.k):
            n = _n_valid(mk)
            starts = [0] if n <= self.W else sorted(set(list(range(0, n - self.W + 1, self.W)) + [n - self.W]))
            self.items += [(i, s) for s in starts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i, s = self.items[j]
        return self.x[i][s:s + self.W], self.m[i], self.y[i][s:s + self.W], self.k[i][s:s + self.W]


def windowed_predict(model, split, window, device, lookahead=None, batch_size=16):
    """Predictions for every trial of `split`, assembled from overlapping windows (hop W/2).
    Returns (preds, actuals, masks) shaped like validate_model's: (N, T_pad, C), (N, T_pad, C), (N, T_pad)."""
    model.eval()
    W, H = int(window), max(1, int(window) // 2)
    X, M, Y, K = split
    preds = np.zeros((len(X),) + tuple(Y[0].shape), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(X)):
            n = _n_valid(K[i])
            if n <= W:
                out = model(X[i][None].to(device), M[i][None].to(device), input_mask=K[i][None].to(device))
                preds[i] = out[0].cpu().numpy(); continue
            starts = sorted(set(list(range(0, n - W + 1, H)) + [n - W]))
            outs = []
            for b in range(0, len(starts), batch_size):
                ss = starts[b:b + batch_size]
                xb = torch.stack([X[i][s:s + W] for s in ss]).to(device)
                mb = M[i][None].expand(len(ss), -1).to(device)
                kb = torch.stack([K[i][s:s + W] for s in ss]).to(device)
                outs.append(model(xb, mb, input_mask=kb).cpu().numpy())
            outs = np.concatenate(outs)
            t = np.arange(n)
            best = np.full(n, -1e9); choice = np.zeros(n, dtype=int)
            for w, s in enumerate(starts):
                inside = (t >= s) & (t < s + W)
                if lookahead is None:
                    score = np.minimum(t - s, s + W - 1 - t).astype(float)
                else:
                    ok = (t <= s + W - 1 - lookahead) | (s + W == n)   # the last window ends at the trial end
                    score = np.where(ok, (t - s).astype(float), -1e8)
                score = np.where(inside, score, -1e9)
                upd = score > best
                best[upd] = score[upd]; choice[upd] = w
            for w, s in enumerate(starts):
                sel = choice == w
                preds[i, t[sel]] = outs[w, t[sel] - s]
    acts = np.stack([y.numpy() for y in Y]); masks = np.stack([k.numpy() for k in K])
    return preds, acts, masks
