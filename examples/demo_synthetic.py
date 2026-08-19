# -*- coding: utf-8 -*-
"""End-to-end synthetic demo of the transition-aware calibration pipeline.

Runs the full method chain on synthetic data -- taxonomy -> severity -> Mondrian conformal
calibration -> metrics -> window-level paired stationary bootstrap -- using only NumPy.
No data, no model checkpoints, and no GPU are required.

This is the fastest way to verify that the method is complete and self-consistent:
    python examples/demo_synthetic.py

Expected behavior:
  - the severity-conditioned (Mondrian) method reduces WCE / GCD relative to the
    unconditional conformal baseline, and
  - the paired-stationary-bootstrap CI for deltaWCE excludes 0.
"""
import os
import sys

import numpy as np

# Make calibration/ and evaluation/ importable when run from anywhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'calibration'))
sys.path.insert(0, os.path.join(_ROOT, 'evaluation'))

import severity as sev        # noqa: E402
import taxonomy as tx         # noqa: E402
import mondrian as cf         # noqa: E402
import bootstrap as bs        # noqa: E402

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
IL, OL = 12, 12


def make_synthetic_data(seed=0):
    """Heteroskedastic traffic-like series with abrupt transition spikes.

    Returns (flow, pred, true, labels) such that:
      - residuals are larger during TypeC windows (heteroskedasticity source),
      - the in-severity r^in of a window is correlated with the residual magnitude.
    """
    rng = np.random.default_rng(seed)
    T, N = 4000, 40
    base = 100.0 + np.cumsum(rng.normal(0, 4, (T, N)), axis=0).clip(min=10)
    # abrupt spikes (sensor/short-term bursts) -> transitions
    for _ in range(120):
        t = rng.integers(20, T - 30)
        n = rng.integers(N)
        base[t:t + rng.integers(2, 6), n] += rng.uniform(80, 300)
    flow = np.maximum(base, 0.0)

    # point forecast = smoothed truth + heteroskedastic noise (same length as flow)
    true = flow
    pred = np.stack([np.convolve(flow[:, n], np.ones(5) / 5, mode='same')
                     for n in range(N)], axis=1)
    # severity_series has length T-1 (one diff per step); extend to length T for broadcasting
    s_series = sev.severity_series(flow, sev.fit_severity_stats(flow[:int(0.6 * T)]))
    sigma = 6.0 + 0.6 * s_series.clip(0, 40)
    sigma = np.concatenate([sigma, sigma[[-1]]], axis=0)          # (T, N)
    pred = pred + rng.normal(0, sigma) * rng.uniform(0.5, 1.5, (T, N))
    return flow, pred, true


def windows(flow, pred, true, g_indices):
    """Stack (n, IL, N) inputs and (n, OL, N) outputs for the given window starts."""
    gs = np.asarray(g_indices)
    x = np.stack([flow[g:g + IL] for g in gs])
    y = np.stack([true[g + IL:g + IL + OL] for g in gs])
    p = np.stack([pred[g + IL:g + IL + OL] for g in gs])
    return x, y, p


def main():
    rng = np.random.default_rng(0)
    flow, pred, true = make_synthetic_data()

    T = flow.shape[0]
    train_end, select_end, calib_end = int(0.6 * T), int(0.7 * T), int(0.8 * T)

    # --- train-only statistics ---
    tau = tx.fit_per_node_q95(flow[:train_end])
    stats = sev.fit_severity_stats(flow[:train_end])
    s = sev.severity_series(flow, stats)

    # --- window construction ---
    g_calib = np.arange(select_end, calib_end - IL - OL + 1)
    g_test = np.arange(calib_end, T - IL - OL + 1)
    xc, yc, pc = windows(flow, pred, true, g_calib)
    xt, yt, pt = windows(flow, pred, true, g_test)

    r_calib = sev.window_in_severity(s, g_calib, IL)
    r_test = sev.window_in_severity(s, g_test, IL)
    labels = tx.classify_from_flow(flow, g_test, tau, IL, OL)
    fc = xc[..., 0].mean(axis=1)
    ft = xt[..., 0].mean(axis=1)
    pmc = np.abs(pc).mean(axis=1)
    pmt = np.abs(pt).mean(axis=1)

    vc = yc > 0
    vt = yt > 0
    n = min(pt.shape[0], labels.shape[0])
    pt, yt, vt = pt[:n], yt[:n], vt[:n]
    labels, r_test, ft, pmt = labels[:n], r_test[:n], ft[:n], pmt[:n]
    pmc = pmc[:pc.shape[0]]

    # --- unconditional conformal baseline ---
    res_u, q = cf.unconditional(pc, yc, pt, yt, labels, ALPHA,
                                valid_val=vc, valid_test=vt)
    print(f'\n== Unconditional conformal ==')
    print(f'  overall_cov={res_u["overall_cov"]:.4f}  WCE={res_u["WCE"]:.4f} '
          f'({res_u["WCE_group"]})  GCD={res_u["GCD"]:.4f}  mean_width={res_u["mean_width"]:.2f}')

    # --- severity-conditioned Mondrian (hard bins) ---
    print(f'\n== Severity-conditioned Mondrian conformal (hard bins) ==')
    res_s, meta_s = cf.mondrian_hard(pc, yc, pt, yt, labels, r_calib, r_test,
                                     K=5, alpha=ALPHA, valid_val=vc, valid_test=vt)
    print(f'  overall_cov={res_s["overall_cov"]:.4f}  WCE={res_s["WCE"]:.4f} '
          f'({res_s["WCE_group"]})  GCD={res_s["GCD"]:.4f}  mean_width={res_s["mean_width"]:.2f} '
          f'  q_monotonic={res_s["q_monotonic"]}')

    # --- flow-level and predicted-magnitude conditioning (scale controls) ---
    res_f, _ = cf.mondrian_hard(pc, yc, pt, yt, labels, fc[:pc.shape[0]], ft,
                                K=5, alpha=ALPHA, valid_val=vc, valid_test=vt)
    res_m, _ = cf.mondrian_hard(pc, yc, pt, yt, labels, pmc, pmt,
                                K=5, alpha=ALPHA, valid_val=vc, valid_test=vt)
    print(f'\n== Scale controls ==')
    for tag, res in [('flow_level', res_f), ('predicted_magnitude', res_m)]:
        print(f'  {tag:20s} WCE={res["WCE"]:.4f}  GCD={res["GCD"]:.4f}')

    # --- per-group coverage table ---
    print(f'\n== Per-group coverage (severity-conditioned) ==')
    for gn in ['Normal', 'TypeA', 'TypeB', 'TypeC']:
        g = res_s[gn]
        print(f'  {gn:8s} cov={g["cov"]:.4f}  n={g["n"]:7d}  width={g["width"]:.2f}')

    # --- window-level paired stationary bootstrap CI for deltaWCE ---
    print(f'\n== Paired stationary bootstrap (deltaWCE = WCE_uncond - WCE_method) ==')
    numU, denU = bs.per_window_counts(pt, yt, q, labels, vt)
    numS, denS = bs.per_window_counts(pt, yt, _q_array(pt, r_test, meta_s), labels, vt)
    (lo, hi, mean), (gl, gh, gm) = bs.paired_bootstrap_delta(
        numU, denU, numS, denS, n_boot=500, seed=42, p=1 / 288, nominal=NOMINAL)
    print(f'  deltaWCE = {mean:+.4f}  CI95 = [{lo:+.4f}, {hi:+.4f}]'
          f'  {"(excludes 0: method better)" if lo > 0 else "(does not exclude 0)"}')
    print(f'  deltaGCD = {gm:+.4f}  CI95 = [{gl:+.4f}, {gh:+.4f}]')
    print('\n[DEMO OK] pipeline ran end-to-end.')


def _q_array(pt, r_test, meta):
    """Rebuild the per-test-window q array from the Mondrian meta (demo helper)."""
    edges = np.array(meta['edges'])
    qb = np.array(meta['q_bins'])
    b = np.clip(np.digitize(np.broadcast_to(r_test[:, None, :], pt.shape), edges[1:-1]), 0, 4)
    return qb[b]


if __name__ == '__main__':
    main()
