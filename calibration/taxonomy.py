# -*- coding: utf-8 -*-
"""Transition taxonomy canonical library -- R1 audit-fixed version (2026-08-17).

Round1 audit fixes:
  P0-2: fixed the 12-step temporal misalignment of the old four-split implementation.
        Labels for window g (input start) are aligned as:
          input diffs  = gdiff[g   : g+IL-1]      (the 11 diffs inside the input window)
          output diffs = gdiff[g+IL: g+IL+OL-1]   (the 11 diffs inside the output window)
        The boundary diff (g+IL-1 -> g+IL, i.e. input end -> output start) is excluded,
        consistent with the legacy np.diff(input)/np.diff(output) semantics.
  P0-3: unified zero mask -- both tau fitting and label assignment use only valid diffs
        (both endpoints flow>0); 0<->x sensor switch-offs are no longer misjudged as transitions.

Label semantics (frozen):
  0=Normal (no abrupt change in either in/out)  1=TypeA (in only)
  2=TypeB (out only)  3=TypeC (in+out)
"""
import numpy as np

NORMAL, TYPE_A, TYPE_B, TYPE_C = 0, 1, 2, 3


def _diffs_masked(flow):
    """|dx| and a validity mask. flow (T,N) -> ad (T-1,N), ok (T-1,N)."""
    x = np.asarray(flow, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=0))
    ok = (x[1:] > 0) & (x[:-1] > 0)
    return ad, ok


def fit_per_node_q95(flow_train):
    """per-node Q95(|dx|), using only valid diffs (P0-3). Returns tau (N,)."""
    ad, ok = _diffs_masked(flow_train)
    _, N = ad.shape
    tau = np.zeros(N)
    for n in range(N):
        v = ad[ok[:, n], n]
        if v.size < 10:
            v = ad[ok]
        tau[n] = np.quantile(v, 0.95)
    return tau


def classify_from_flow(flow, g_indices, tau, input_len=12, output_len=12):
    """Raw full-series version (four-split STAEformer/BasicTS path).

    flow (T,N); g_indices = input start g of each window. Returns labels (n_win, N) int8.
    input diffs [g, g+IL-1); output diffs [g+IL, g+IL+OL-1); the boundary diff is excluded.
    """
    ad, ok = _diffs_masked(flow)
    g = np.asarray(g_indices)
    n_win = len(g)
    labels = np.zeros((n_win, flow.shape[1]), dtype=np.int8)
    ar_in = np.arange(input_len - 1)
    ar_out = np.arange(output_len - 1)
    for i, gi in enumerate(g):
        # (IL-1,N) / (OL-1,N); out-of-range windows are the caller's responsibility
        hi = ((ad[gi + ar_in] > tau) & ok[gi + ar_in]).any(axis=0)
        ho = ((ad[gi + input_len + ar_out] > tau) & ok[gi + input_len + ar_out]).any(axis=0)
        labels[i, hi & ~ho] = TYPE_A
        labels[i, ~hi & ho] = TYPE_B
        labels[i, hi & ho] = TYPE_C
    return labels


def classify_from_windows(x_win, y_win, tau):
    """Legacy npz window version (UC-G1 three-split / STID path; semantics already correct,
    zero mask added here).

    x_win (n,IL,N), y_win (n,OL,N). np.diff inside the windows; no boundary diff exists.
    """
    fx = np.asarray(x_win, dtype=np.float64)
    fy = np.asarray(y_win, dtype=np.float64)
    gin = np.abs(np.diff(fx, axis=1)) > tau[None, None, :]
    vin = (fx[:, 1:] > 0) & (fx[:, :-1] > 0)
    gin &= vin
    gout = np.abs(np.diff(fy, axis=1)) > tau[None, None, :]
    vout = (fy[:, 1:] > 0) & (fy[:, :-1] > 0)
    gout &= vout
    hi = gin.any(axis=1); ho = gout.any(axis=1)
    labels = np.zeros((fx.shape[0], fx.shape[2]), dtype=np.int8)
    labels[hi & ~ho] = TYPE_A
    labels[~hi & ho] = TYPE_B
    labels[hi & ho] = TYPE_C
    return labels
