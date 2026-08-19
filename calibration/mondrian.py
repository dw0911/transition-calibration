# -*- coding: utf-8 -*-
"""Conformal calibration canonical library -- Round-1 audit fix (R1, 2026-08-17).

Fixes corresponding to audit items:
  P0-5: mondrian_hard (main method, standard Mondrian in-group guarantee) and
        mondrian_interpolated (smooth variant, engineering enhancement, no standard
        finite-sample guarantee) are explicitly separated.
  P0-6: conformal_quantile implements the pre-registered finite-sample correction
        k=ceil((n+1)(1-alpha)) (method="higher").
  P0-3: valid mask (true>0) is threaded through residual quantile / coverage / width / WCE / GCD.
  P1-4: q-monotonicity check records q_monotonic metadata (does not force a change).

Conventions (library-wide):
  - conformal score unit: node-horizon (window, horizon, node)
  - coverage / WCE / GCD computed only on valid_target (default true>0)
  - residual quantile uses only valid residuals
  - reported as empirical node-horizon coverage (no guarantee claim given temporal/spatial dependence)
"""
import numpy as np

# ---------------------------------------------------------------------------
# P0-6: pre-registered finite-sample corrected quantile
# ---------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    """Split conformal quantile: the k=ceil((n+1)(1-alpha))-th order statistic of scores
    (ascending, 1-based).

    When k>n (small-sample alpha finer than 1/n), it shrinks to n (take the maximum, the
    correct finite-sample behavior). Uses direct ranked indexing to avoid the off-by-one of
    np.quantile linear interpolation (h=(n-1)*level with 'higher' points to rank+1, caught by T8b test).
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        raise ValueError('conformal_quantile: empty scores')
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    return float(np.sort(scores)[k - 1])


# ---------------------------------------------------------------------------
# Evaluation (P0-3: valid mask threaded through)
# ---------------------------------------------------------------------------

def _eval_groups(pred, true, q, labels, valid_target=None, nominal=0.9):
    """Per-group coverage/width/gap + WCE/GCD.

    pred/true: (n,H,N); q: scalar or (n,H,N); labels: (n,N) int8;
    valid_target: (n,H,N) bool, None means all valid (caller owns the true>0 semantics).
    """
    covered = (true >= pred - q) & (true <= pred + q)
    width = 2 * (q if np.isscalar(q) else q)
    if valid_target is None:
        valid_target = np.ones(covered.shape, dtype=bool)
    else:
        valid_target = np.asarray(valid_target, dtype=bool)
    n_valid_all = int(valid_target.sum())
    out = {'n_valid': n_valid_all, 'n_total': int(valid_target.size)}
    if n_valid_all > 0:
        out['overall_cov'] = float((covered & valid_target).sum() / n_valid_all)
        out['mean_width'] = float((width * valid_target).sum() / n_valid_all)
    else:
        out['overall_cov'] = float('nan')
        out['mean_width'] = float('nan')
    cn = None
    for tid, gn in {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}.items():
        sel = (labels == tid)[:pred.shape[0]]                      # (n,N)
        gmask = np.broadcast_to(np.expand_dims(sel, 1), covered.shape) & valid_target
        n = int(gmask.sum())
        if n == 0:
            out[gn] = {'cov': float('nan'), 'width': float('nan'), 'n': 0}
            continue
        cov = float((covered & gmask).sum() / n)
        gw = float(width) if np.isscalar(q) else float((width * gmask).sum() / n)
        out[gn] = {'cov': cov, 'gap': cov - nominal, 'width': gw, 'n': n}
        if tid == 0:
            cn = cov
    if cn is not None:
        for gn in ['TypeA', 'TypeB', 'TypeC']:
            if gn in out and not np.isnan(out[gn]['cov']):
                out[gn]['cond_gap'] = out[gn]['cov'] - cn
    covs = [out[g]['cov'] for g in ['Normal', 'TypeA', 'TypeB', 'TypeC']
            if g in out and not np.isnan(out[g]['cov'])]
    if covs:
        out['WCE'] = float(max(abs(c - nominal) for c in covs))
        out['GCD'] = float(max(covs) - min(covs))
        # WCE responsible group (audit item P0-C)
        gaps = {g: abs(out[g]['cov'] - nominal) for g in ['Normal', 'TypeA', 'TypeB', 'TypeC']
                if g in out and not np.isnan(out[g]['cov'])}
        out['WCE_group'] = max(gaps, key=gaps.get)
    return out


# ---------------------------------------------------------------------------
# Unconditional
# ---------------------------------------------------------------------------

def unconditional(pred_val, true_val, pred_test, true_test, labels, alpha=0.1,
                  valid_val=None, valid_test=None):
    """Global conformal (estimate q from valid residuals). Returns (res, q)."""
    nominal = 1.0 - alpha
    R = np.abs(pred_val - true_val)
    if valid_val is not None:
        R = R[np.asarray(valid_val, dtype=bool)]
    q = conformal_quantile(R.ravel(), alpha)
    return _eval_groups(pred_test, true_test, q, labels, valid_test, nominal), q


# ---------------------------------------------------------------------------
# Mondrian (P0-5: hard main method / interpolated variant)
# ---------------------------------------------------------------------------

def _broadcast_cond(cond, shape):
    """cond (n,) or (n,N) -> (n,H,N)."""
    cond = np.asarray(cond)
    if cond.ndim == 1:
        return np.broadcast_to(cond[:, None, None], shape)
    return np.broadcast_to(cond[:, None, :], shape)


def _bin_fit(pred_val, true_val, cond_val, K, alpha, valid_val=None):
    """Bin fitting: edges / anchors / q_bins (per-bin conformal_quantile) / monotonicity flag."""
    R = np.abs(pred_val - true_val)                                # (n,H,N)
    cv = _broadcast_cond(cond_val, R.shape)
    if valid_val is not None:
        vm = np.asarray(valid_val, dtype=bool)
    else:
        vm = np.ones(R.shape, dtype=bool)
    Rf, cvf = R[vm], cv[vm]
    if Rf.size == 0:
        raise ValueError('_bin_fit: no valid calibration elements')
    edges = np.quantile(cvf, np.linspace(0, 1, K + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    bidx = np.digitize(cvf, edges[1:-1])                           # 0..K-1
    gq = conformal_quantile(Rf, alpha)                             # small-bin fallback
    q_bins, anchors = np.zeros(K), np.zeros(K)
    for k in range(K):
        m = bidx == k
        q_bins[k] = conformal_quantile(Rf[m], alpha) if int(m.sum()) > 10 else gq
        anchors[k] = float(np.median(cvf[m])) if int(m.sum()) > 0 else float(edges[k])
    order = np.argsort(anchors)
    anchors, q_bins = anchors[order], q_bins[order]
    mono = bool(np.all(np.diff(q_bins) >= -1e-12))                 # P1-4: check, do not alter
    return edges, anchors, q_bins, mono


def mondrian_hard(pred_val, true_val, pred_test, true_test, labels, cond_val, cond_test,
                  K=5, alpha=0.1, valid_val=None, valid_test=None):
    """Main method: hard-bin Mondrian conformal.

    Each bin k uses its in-bin conformal_quantile(n_k); a test point takes q_k of the bin its
    conditioning variable falls into (no interpolation, no extrapolation -- values beyond edges
    are clamped to the first/last bin, which is within the bin definition).
    Returns (res, meta); meta contains edges/anchors/q_bins/q_monotonic/per-bin n.
    """
    nominal = 1.0 - alpha
    edges, anchors, q_bins, mono = _bin_fit(pred_val, true_val, cond_val, K, alpha, valid_val)
    ct = _broadcast_cond(cond_test, pred_test.shape)
    b = np.clip(np.digitize(ct, edges[1:-1]), 0, K - 1)
    q_test = q_bins[b]
    res = _eval_groups(pred_test, true_test, q_test, labels, valid_test, nominal)
    res['q_monotonic'] = mono
    meta = {'edges': edges.tolist(), 'anchors': anchors.tolist(),
            'q_bins': q_bins.tolist(), 'q_monotonic': mono, 'K': K}
    return res, meta


def mondrian_interpolated(pred_val, true_val, pred_test, true_test, labels, cond_val, cond_test,
                          K=5, alpha=0.1, valid_val=None, valid_test=None):
    """Smooth variant: linear interpolation across adjacent bin anchors (np.interp, no extrapolation).

    Note: the interpolated q no longer equals the formal conformal quantile of the bin it falls in,
    so the standard Mondrian finite-sample guarantee is not automatically preserved -- reported
    as an engineering enhancement only.
    """
    nominal = 1.0 - alpha
    edges, anchors, q_bins, mono = _bin_fit(pred_val, true_val, cond_val, K, alpha, valid_val)
    ct = _broadcast_cond(cond_test, pred_test.shape)
    q_test = np.interp(ct, anchors, q_bins)
    res = _eval_groups(pred_test, true_test, q_test, labels, valid_test, nominal)
    res['q_monotonic'] = mono
    meta = {'edges': edges.tolist(), 'anchors': anchors.tolist(),
            'q_bins': q_bins.tolist(), 'q_monotonic': mono, 'K': K}
    return res, meta


# ---------------------------------------------------------------------------
# Three-color gate (project-management gate, not a paper metric -- retained from audit P0)
# ---------------------------------------------------------------------------

def three_color(res_method, res_uncond):
    cov_tc = res_method['TypeC']['cov']
    cov_uncond_tc = res_uncond['TypeC']['cov']
    gain = cov_tc - cov_uncond_tc
    width_ratio = res_method['mean_width'] / res_uncond['mean_width']
    if (cov_tc >= 0.90 or gain >= 0.10) and width_ratio <= 1.5:
        color = 'Green'
    elif 0.05 <= gain < 0.10 and width_ratio <= 1.3:
        color = 'Yellow'
    else:
        color = 'Red'
    return {'color': color, 'gain': gain, 'width_ratio': width_ratio,
            'cov_TypeC': cov_tc, 'cov_TypeC_uncond': cov_uncond_tc}
