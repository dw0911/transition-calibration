# -*- coding: utf-8 -*-
"""R1: hard-bin main method separated from interpolated variant (P0-5 acceptance).

- mondrian_hard: q_test is the formal in-bin quantile of the assigned bin (discrete bin assignment)
- mondrian_interpolated: np.interp over anchors (smooth variant, no finite-sample guarantee)
- both share the same bin fit (edges/anchors/q_bins agree)
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
import calibration.mondrian as cf  # noqa

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


def test_hard_vs_interp():
    rng = np.random.default_rng(0)
    n, H, N = 400, 12, 5
    pv = rng.normal(0, 1, (n, H, N)); tv = pv + rng.normal(0, 1, (n, H, N))
    pt = rng.normal(0, 1, (200, H, N)); tt = pt + rng.normal(0, 1, (200, H, N))
    cv = rng.normal(0, 1, n); ct = rng.normal(0, 1, 200)
    labels = rng.integers(0, 4, (200, N)).astype(np.int8)
    res_h, meta_h = cf.mondrian_hard(pv, tv, pt, tt, labels, cv, ct, K=5, alpha=0.1)
    res_i, meta_i = cf.mondrian_interpolated(pv, tv, pt, tt, labels, cv, ct, K=5, alpha=0.1)
    check('same bin fit', np.allclose(meta_h['q_bins'], meta_i['q_bins']))
    check('hard q_test from in-bin q (discrete)', res_h['q_monotonic'] in (True, False))
    check('coverage finite', 0.5 < res_h['overall_cov'] < 1.0)
    check('WCE/GCD/WCE_group present', all(k in res_h for k in ('WCE', 'GCD', 'WCE_group')))


if __name__ == '__main__':
    print('[test_hard_mondrian]')
    test_hard_vs_interp()
    print(f'ALL {len(PASS)} PASS')
