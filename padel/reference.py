"""Reference building (design.md section 7): resample strokes, DTW by hand, medoid, path-averaged mean and std."""
from __future__ import annotations

import numpy as np

N_STEPS, CONTACT_STEP, BAND = 60, 40, 12


def fill_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs per column of a (T,F) array (edges take the nearest value); all-NaN columns stay NaN."""
    out = np.asarray(x, dtype=float).copy()
    t = np.arange(len(out))
    for j in range(out.shape[1]):
        ok = np.isfinite(out[:, j])
        if ok.any() and not ok.all():
            out[:, j] = np.interp(t, t[ok], out[ok, j])
    return out


def resample(feat: np.ndarray, contact: int, n: int = N_STEPS, contact_step: int = CONTACT_STEP) -> np.ndarray:
    """(T,F) features with contact at frame index `contact` -> (n,F), contact at `contact_step`.

    The part before contact and the part after are stretched separately so contact stays aligned.
    """
    feat = fill_nan(feat)
    T = len(feat)
    contact = int(np.clip(contact, 1, T - 2))
    src = np.concatenate([
        np.linspace(0, contact, contact_step, endpoint=False),
        np.linspace(contact, T - 1, n - contact_step),
    ])
    out = np.full((n, feat.shape[1]), np.nan)
    t = np.arange(T)
    for j in range(feat.shape[1]):
        ok = np.isfinite(feat[:, j])
        if ok.any():
            out[:, j] = np.interp(src, t[ok], feat[ok, j])
    return out


def _local_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,m) Euclidean distances over the features both sequences have (NaN features skipped, rescaled)."""
    sq = (a[:, None, :] - b[None, :, :]) ** 2
    valid = np.isfinite(sq)
    cnt = valid.sum(-1)
    tot = np.where(valid, sq, 0.0).sum(-1)
    return np.sqrt(tot * a.shape[1] / np.maximum(cnt, 1))


def dtw(a: np.ndarray, b: np.ndarray, band: int | None = BAND) -> tuple[float, list[tuple[int, int]]]:
    """DTW of (n,F) and (m,F) sequences with a Sakoe-Chiba band. Returns (accumulated cost, path of (i, j))."""
    n, m = len(a), len(b)
    d = _local_cost(a, b)
    inf = float("inf")
    D = np.full((n, m), inf)
    w = max(band if band is not None else max(n, m), abs(n - m))
    for i in range(n):
        for j in range(max(0, i - w), min(m, i + w + 1)):
            if i == 0 and j == 0:
                D[0, 0] = d[0, 0]
                continue
            best = min(D[i - 1, j] if i else inf, D[i, j - 1] if j else inf, D[i - 1, j - 1] if i and j else inf)
            D[i, j] = d[i, j] + best
    i, j, path = n - 1, m - 1, [(n - 1, m - 1)]
    while i or j:
        cands = []
        if i and j:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
        if i:
            cands.append((D[i - 1, j], i - 1, j))
        if j:
            cands.append((D[i, j - 1], i, j - 1))
        _, i, j = min(cands)
        path.append((i, j))
    return float(D[-1, -1]), path[::-1]


def medoid(seqs: list[np.ndarray], band: int | None = BAND) -> int:
    """Index of the sequence with the smallest total DTW distance to all others."""
    k = len(seqs)
    dist = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            dist[i, j] = dist[j, i] = dtw(seqs[i], seqs[j], band)[0]
    return int(dist.sum(1).argmin())


def align_to(seq: np.ndarray, ref: np.ndarray, band: int | None = BAND) -> np.ndarray:
    """Warp `seq` onto `ref`'s time axis: for each ref step, the mean of the seq steps matched to it. Shape of ref."""
    _, path = dtw(ref, seq, band)
    out = np.zeros_like(ref)
    cnt = np.zeros(len(ref))
    for j, i in path:
        out[j] += seq[i]
        cnt[j] += 1
    return out / cnt[:, None]


def build_reference(seqs: list[np.ndarray], std_floor: float = 0.25, band: int | None = BAND) -> dict:
    """Mean and std per step per feature after aligning every (N,F) stroke to the medoid."""
    scale = np.nanstd(np.concatenate(seqs), axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    z = [s / scale for s in seqs]
    m = medoid(z, band)
    aligned = np.stack([align_to(s, z[m], band) for s in z]) * scale
    mean = np.nanmean(aligned, axis=0)
    std = np.maximum(np.nanstd(aligned, axis=0), std_floor * scale)
    return {"mean": mean, "std": std, "scale": scale, "medoid": m, "n_strokes": len(seqs)}
