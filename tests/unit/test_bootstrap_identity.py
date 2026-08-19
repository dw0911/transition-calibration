# -*- coding: utf-8 -*-
"""R1: window-level paired stationary bootstrap (P0-7 acceptance).

- Resampling unit = forecast window (not flat elements)
- original-sample estimate (direct point estimate) is consistent with the bootstrap mechanism
- bootstrap original-sample identity: direct deltaWCE == deltaWCE from all-window pooled counts
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
import evaluation.bootstrap as bs  # noqa

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


def test_identity_and_window_unit():
    """Setup: TypeC group has large residuals (sigma=20), Normal group small (sigma=5), heteroskedastic.

    - uncond uses global q=Q0.9(all residuals) -> Normal over-covers / TypeC under-covers -> large WCE
    - method uses per-group q_k=Q0.9(in-group residuals) -> each group cov~=0.90 -> WCE~=0
    -> deltaWCE = WCE_uncond - WCE_method significantly > 0, bootstrap CI should exclude 0.
    """
    rng = np.random.default_rng(7)
    n, H, N = 300, 12, 10
    true = rng.uniform(20, 200, (n, H, N))
    node_group = rng.integers(0, 4, N).astype(np.int8)       # fixed node attribute (heteroskedasticity source)
    labels = np.tile(node_group[None, :], (n, 1)).astype(np.int8)  # (n,N)
    sigma = np.where(node_group == 3, 20.0, 5.0)[None, None, :]  # (1,1,N)
    err = rng.normal(0, 1, (n, H, N)) * sigma
    pred = true + err
    vt = true > 0
    # per-window per-group residual -> global q / per-group q (data-driven, not a fixed constant)
    R = np.abs(pred - true)
    R_valid = np.where(vt, R, np.nan)
    q_global = float(np.nanquantile(R_valid, 0.9))
    q_grp = np.array([np.nanquantile(R_valid[:, :, node_group == g], 0.9)
                      for g in range(4)])                  # (4,)
    # uncond: global q; method: per-node-group q (true Mondrian semantics)
    qU = np.full_like(pred, q_global)
    qM = q_grp[node_group][None, None, :] * np.ones((n, H, N))
    numU, denU = bs.per_window_counts(pred, true, qU, labels, vt)
    numM, denM = bs.per_window_counts(pred, true, qM, labels, vt)
    # original-sample identity
    dw_direct, dg_direct = bs.original_sample_estimate(numU, denU, numM, denM)
    wu, gu, _ = bs.wce_gcd_from_pooled(numU.sum(0), denU.sum(0))
    wm, gm, _ = bs.wce_gcd_from_pooled(numM.sum(0), denM.sum(0))
    check('identity direct == pooled', abs(dw_direct - (wu - wm)) < 1e-12)
    # bootstrap CI around point estimate and excludes 0 (method's group-level calibration is significantly better)
    (dl, dh, dm), (gl, gh, gm2) = bs.paired_bootstrap_delta(
        numU, denU, numM, denM, n_boot=500, seed=42, p=1/288)
    check('CI around point', abs(dm - dw_direct) < 0.02)
    check('deltaWCE CI excludes 0 (method better)', dl > 0)
    check('deltaGCD CI excludes 0', gl > 0)
    check('window unit (num shape (n,4))', numU.shape == (n, 4))


if __name__ == '__main__':
    print('[test_bootstrap_identity]')
    test_identity_and_window_unit()
    print(f'ALL {len(PASS)} PASS')
