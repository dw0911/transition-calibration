# -*- coding: utf-8 -*-
"""Transition severity construction module -- core asset of the UC study (R1 audit-fixed version, 2026-08-17).

Frozen formulas (uncalib_v1/PLAN.md + hypotheses/UC-H2.md):

    severity:      s_{n,t} = |dx_{n,t} - m_n| / d_n
                   m_n = median(|dx_n|),  d_n = 1.4826 * MAD_n     (train-only statistics)
    window severity:
                   r_{i,n}^in  = max_{t in W_in(i)}  s_{n,t}   (past-only, UC-G2 conditioning)
                   r_{i,n}^out = max_{t in W_out(i)} s_{n,t}   (used by Task E; UC diagnosis control only)

Engineering constraints (frozen, fully enforced since R1):
  - zero mask: if either endpoint of a difference has flow=0, that difference does not enter the
    statistics and is set to 0 in the series
  - 99th-percentile clip applied before: s <- min(s, Q99(s_train)) (P1-1 fix: code aligned with the
    protocol; r = max over window <= Q99 inherits naturally, equivalent to the protocol's r <- min(r, Q99(r_train)))
  - tau / clip quantiles are taken from train statistics; val/test only apply, never re-estimate (no leakage)

R1 fixes (Round1 audit):
  - P1-3: fit_severity_stats fallback indexing changed to ad[ok] (correct 2D-bool semantics)
  - P1-1: Q99 clip actually implemented (previously only declared in docs)
  - P1-2 note: s is a two-sided deviation |dx|-m (change-magnitude anomaly severity; the frozen
    definition is kept; one-sided sensitivity control lives in the ucg1c family)

Deployment boundary (UC-G2 frozen): the interval width depends only on the historically observable
r^in; inference never reads future TypeC/severity.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeverityStats:
    m: np.ndarray      # (N,) per-node median of |dx| (train, zero-masked)
    d: np.ndarray      # (N,) per-node 1.4826*MAD of |dx| (train, zero-masked)
    q99: float = 0.0   # 99th percentile of valid s over train (clip upper bound, P1-1)

    def asdict(self):
        return {'m': self.m.tolist(), 'd': self.d.tolist(), 'q99': self.q99}


def _absdiff_masked(flow):
    """|dx| and a validity mask. flow (T,N) -> ad (T-1,N), ok (T-1,N) bool."""
    x = np.asarray(flow, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=0))
    ok = (x[1:] > 0) & (x[:-1] > 0)
    return ad, ok


def fit_severity_stats(flow_train):
    """Fit per-node robust statistics (zero-masked) plus the global Q99 clip bound on train."""
    ad, ok = _absdiff_masked(flow_train)
    _, N = ad.shape
    m = np.zeros(N); d = np.ones(N)
    for n in range(N):
        v = ad[ok[:, n], n]                       # valid diffs of this node
        if v.size < 10:
            v = ad[ok]                            # P1-3 fix: global valid fallback (2D bool index)
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        m[n] = med
        d[n] = 1.4826 * mad if mad > 0 else max(np.std(v), 1e-6)
    stats = SeverityStats(m=m, d=d)
    # P1-1: Q99 of valid s over train (clip upper bound; zero positions have s=0 and do not enter the quantile)
    s_train = np.where(ok, np.abs(ad - m) / d, 0.0)
    valid_s = s_train[ok]
    stats.q99 = float(np.quantile(valid_s, 0.99)) if valid_s.size > 0 else float('inf')
    return stats


def severity_series(flow, stats):
    """Full-series severity s (T-1, N). Zero diffs set to 0; P1-1: clipped to the train Q99."""
    ad, ok = _absdiff_masked(flow)
    s_raw = np.abs(ad - stats.m) / stats.d
    s = np.where(ok, np.minimum(s_raw, stats.q99), 0.0)   # zero positions -> 0 + Q99 clip
    return s


def window_in_severity(s, g_indices, input_len=12):
    """r^in (n_win, N): W_in diff indices [g, g+input_len-1). g = input start."""
    g = np.asarray(g_indices)
    ar = np.arange(input_len - 1)
    blk = s[g[:, None] + ar] if len(g) else s[:0]    # (n_win, IN-1, N)
    return blk.max(axis=1)


def window_out_severity(s, p_indices, output_len=12):
    """r^out (n_win, N): W_out diff indices [p, p+output_len-1). p = output start."""
    p = np.asarray(p_indices)
    ar = np.arange(output_len - 1)
    blk = s[p[:, None] + ar] if len(p) else s[:0]
    return blk.max(axis=1)


def window_in_severity_from_windows(x_win, stats, input_len=12):
    """Window-array version of r^in (legacy npz path, no series anchor needed).

    x_win: (n, input_len, N) raw flow. Strictly equivalent to window_in_severity on the same
    windows (r^in depends only on the in-window diffs). P0-4 fix: the STID windowed-data path
    reuses the same severity definition.
    """
    x = np.asarray(x_win, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=1))                     # (n, IL-1, N)
    ok = (x[:, 1:] > 0) & (x[:, :-1] > 0)
    s = np.where(ok, np.minimum(np.abs(ad - stats.m) / stats.d, stats.q99), 0.0)
    return s.max(axis=1)
