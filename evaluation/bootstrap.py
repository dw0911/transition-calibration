# -*- coding: utf-8 -*-
"""Stationary bootstrap canonical module -- window-level paired method-effect CI (P0-7 correct implementation).

Corresponding audit items:
  R1 bootstrap must use the forecast window as the resampling unit
  R1 recompute WCE/GCD from pooled within-group numerator/denominator
  deltaWCE = WCE_uncond - WCE_method; deltaGCD analogously
  iid bootstrap over flat node-horizon elements is forbidden

Reference: the R4 implementation (R2_recalc/r4_bootstrap.py) factored out into a common module.
"""
import numpy as np


def per_window_counts(pred, true, q, labels, valid_target=None):
    """Per-window covered num / valid den (per group, 4 classes).

    pred/true: (n,H,N); q: scalar or (n,H,N); labels: (n,N) int8.
    Returns num, den: (n,4) float64.
    """
    vt = np.ones(pred.shape, dtype=bool) if valid_target is None else \
        np.asarray(valid_target, dtype=bool)
    covered = (true >= pred - q) & (true <= pred + q) & vt
    n = pred.shape[0]
    num = np.zeros((n, 4), dtype=np.float64)
    den = np.zeros((n, 4), dtype=np.float64)
    for tid in range(4):
        m = np.broadcast_to((labels == tid)[:, None, :], covered.shape) & vt
        num[:, tid] = (covered & m).sum(axis=(1, 2))
        den[:, tid] = m.sum(axis=(1, 2))
    return num, den


def wce_gcd_from_pooled(num, den, nominal=0.9):
    """pooled counts -> (WCE, GCD, WCE_group). num/den: (4,) or the sum of (n,4)."""
    covs = np.full(4, np.nan)
    for tid in range(4):
        if den[tid] > 0:
            covs[tid] = num[tid] / den[tid]
    wce = float(np.nanmax(np.abs(covs - nominal)))
    gcd = float(np.nanmax(covs) - np.nanmin(covs))
    gaps = {i: abs(covs[i] - nominal) for i in range(4) if not np.isnan(covs[i])}
    wg = int(max(gaps, key=gaps.get))
    return wce, gcd, wg


def stationary_bootstrap_resample(n_win, rng, p):
    """Stationary-bootstrap window indices (Politis-Romano random blocks)."""
    idx = np.empty(n_win, dtype=np.int64)
    idx[0] = rng.integers(n_win)
    for i in range(1, n_win):
        idx[i] = rng.integers(n_win) if rng.random() < p else (idx[i - 1] + 1) % n_win
    return idx


def paired_bootstrap_delta(num_uncond, den_uncond, num_method, den_method,
                           n_boot=1000, seed=42, p=1.0 / 288, nominal=0.9):
    """Window-level paired stationary bootstrap CI for deltaWCE / deltaGCD.

    num_uncond/den_uncond: (n,4) per-window uncond counts; num_method/den_method likewise.
    Returns ((dw_lo, dw_hi, dw_mean), (dg_lo, dg_hi, dg_mean)).
    """
    rng = np.random.default_rng(seed)
    n_win = num_uncond.shape[0]
    dw = np.empty(n_boot)
    dg = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_resample(n_win, rng, p)
        nu, du = num_uncond[idx].sum(axis=0), den_uncond[idx].sum(axis=0)
        nm, dm = num_method[idx].sum(axis=0), den_method[idx].sum(axis=0)
        wu, gu, _ = wce_gcd_from_pooled(nu, du, nominal)
        wm, gm, _ = wce_gcd_from_pooled(nm, dm, nominal)
        dw[b] = wu - wm
        dg[b] = gu - gm
    dwci = (float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5)), float(dw.mean()))
    dgci = (float(np.percentile(dg, 2.5)), float(np.percentile(dg, 97.5)), float(dg.mean()))
    return dwci, dgci


def original_sample_estimate(num_uncond, den_uncond, num_method, den_method, nominal=0.9):
    """Bootstrap original-sample identity check: direct pooled estimate within the replicate."""
    nu, du = num_uncond.sum(axis=0), den_uncond.sum(axis=0)
    nm, dm = num_method.sum(axis=0), den_method.sum(axis=0)
    wu, gu, _ = wce_gcd_from_pooled(nu, du, nominal)
    wm, gm, _ = wce_gcd_from_pooled(nm, dm, nominal)
    return wu - wm, gu - gm
