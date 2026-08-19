# -*- coding: utf-8 -*-
"""End-to-end regression: the full transition-aware calibration pipeline on synthetic data.

This test mirrors `examples/demo_synthetic.py` and asserts the qualitative claims that make
the method meaningful:
  - severity-conditioned Mondrian conformal reduces WCE / GCD vs the unconditional baseline,
  - the paired stationary bootstrap CI for deltaWCE excludes 0,
  - per-group coverage exists for all four transition classes.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT

import calibration.severity as sev            # noqa: E402
import calibration.taxonomy as tx             # noqa: E402
import calibration.mondrian as cf            # noqa: E402
import evaluation.bootstrap as bs            # noqa: E402

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
IL, OL = 12, 12


def _make_synthetic_data(seed=0):
    rng = np.random.default_rng(seed)
    T, N = 4000, 40
    base = 100.0 + np.cumsum(rng.normal(0, 4, (T, N)), axis=0).clip(min=10)
    for _ in range(120):
        t = rng.integers(20, T - 30)
        n = rng.integers(N)
        base[t:t + rng.integers(2, 6), n] += rng.uniform(80, 300)
    flow = np.maximum(base, 0.0)

    pred = np.stack([np.convolve(flow[:, n], np.ones(5) / 5, mode='same')
                     for n in range(N)], axis=1)
    s_series = sev.severity_series(flow, sev.fit_severity_stats(flow[:int(0.6 * T)]))
    sigma = 6.0 + 0.6 * s_series.clip(0, 40)
    sigma = np.concatenate([sigma, sigma[[-1]]], axis=0)
    pred = pred + rng.normal(0, sigma) * rng.uniform(0.5, 1.5, (T, N))
    return flow, pred


def _q_array(pt, r_test, meta):
    edges = np.array(meta['edges'])
    qb = np.array(meta['q_bins'])
    b = np.clip(np.digitize(np.broadcast_to(r_test[:, None, :], pt.shape), edges[1:-1]), 0, 4)
    return qb[b]


def test_demo_pipeline():
    flow, pred = _make_synthetic_data()
    true = flow
    T = flow.shape[0]
    train_end, select_end, calib_end = int(0.6 * T), int(0.7 * T), int(0.8 * T)

    tau = tx.fit_per_node_q95(flow[:train_end])
    stats = sev.fit_severity_stats(flow[:train_end])
    s = sev.severity_series(flow, stats)

    g_calib = np.arange(select_end, calib_end - IL - OL + 1)
    g_test = np.arange(calib_end, T - IL - OL + 1)
    xc = np.stack([flow[g:g + IL] for g in g_calib]); yc = np.stack([true[g + IL:g + IL + OL] for g in g_calib])
    pc = np.stack([pred[g + IL:g + IL + OL] for g in g_calib])
    xt = np.stack([flow[g:g + IL] for g in g_test]); yt = np.stack([true[g + IL:g + IL + OL] for g in g_test])
    pt = np.stack([pred[g + IL:g + IL + OL] for g in g_test])

    r_calib = sev.window_in_severity(s, g_calib, IL)
    r_test = sev.window_in_severity(s, g_test, IL)
    labels = tx.classify_from_flow(flow, g_test, tau, IL, OL)
    fc = xc[..., 0].mean(axis=1); ft = xt[..., 0].mean(axis=1)
    pmc = np.abs(pc).mean(axis=1); pmt = np.abs(pt).mean(axis=1)

    vc = yc > 0; vt = yt > 0
    n = min(pt.shape[0], labels.shape[0])
    pt, yt, vt = pt[:n], yt[:n], vt[:n]
    labels, r_test, ft, pmt = labels[:n], r_test[:n], ft[:n], pmt[:n]

    res_u, q = cf.unconditional(pc, yc, pt, yt, labels, ALPHA, valid_val=vc, valid_test=vt)
    res_s, meta_s = cf.mondrian_hard(pc, yc, pt, yt, labels, r_calib, r_test,
                                     K=5, alpha=ALPHA, valid_val=vc, valid_test=vt)
    res_f, _ = cf.mondrian_hard(pc, yc, pt, yt, labels, fc[:pc.shape[0]], ft,
                                K=5, alpha=ALPHA, valid_val=vc, valid_test=vt)
    res_m, _ = cf.mondrian_hard(pc, yc, pt, yt, labels, pmc, pmt,
                                K=5, alpha=ALPHA, valid_val=vc, valid_test=vt)

    # qualitative claims
    assert res_u['overall_cov'] > 0.7, 'unconditional coverage too low'
    assert res_s['WCE'] <= res_u['WCE'], 'severity conditioning should not increase WCE'
    assert res_s['GCD'] <= res_u['GCD'], 'severity conditioning should not increase GCD'
    for gn in ['Normal', 'TypeA', 'TypeB', 'TypeC']:
        assert res_s[gn]['n'] > 0, f'{gn} group empty in severity-conditioned eval'

    # paired stationary bootstrap: severity improves over uncond
    numU, denU = bs.per_window_counts(pt, yt, q, labels, vt)
    numS, denS = bs.per_window_counts(pt, yt, _q_array(pt, r_test, meta_s), labels, vt)
    (lo, hi, mean), _ = bs.paired_bootstrap_delta(
        numU, denU, numS, denS, n_boot=200, seed=42, p=1 / 288, nominal=NOMINAL)
    assert lo > 0, 'deltaWCE CI must exclude 0 (method better)'
    assert hi > mean > 0
    print(f'  deltaWCE={mean:+.4f} CI=[{lo:+.4f},{hi:+.4f}] OK')
