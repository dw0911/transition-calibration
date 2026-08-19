# -*- coding: utf-8 -*-
"""R1: valid mask end-to-end consistency (P0-3 acceptance).

Verifies:
  - valid_target computes coverage/width only where true>0 AND finite
  - sum_g count_g = count_all_valid (mask conservation)
  - residual quantile uses only valid residuals
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
sys.path.insert(0, os.path.join(UC, 'SRC'))
import conformal as cf  # noqa
import evaluation_protocol as ep  # noqa

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


def test_mask_conservation():
    """sum_g count_g = count_all_valid."""
    rng = np.random.default_rng(0)
    n, H, N = 50, 12, 8
    pred = rng.normal(50, 10, (n, H, N))
    true = rng.uniform(0, 200, (n, H, N)); true[::7, :, ::3] = 0.0  # punch holes
    labels = rng.integers(0, 4, (n, N)).astype(np.int8)
    vt = ep.compute_valid_target(true)
    res = cf._eval_groups(pred, true, 20.0, labels, vt, 0.9)
    s = sum(res[g]['n'] for g in ['Normal', 'TypeA', 'TypeB', 'TypeC'])
    check('mask conservation sum_g count_g==count_all_valid', s == res['n_valid'])
    check('valid only where true>0', res['n_valid'] == int(vt.sum()))


def test_target_mask():
    """All valid positions covered -> cov=1; zeros excluded from denominator."""
    pt = np.full((10, 12, 4), 50.0)
    tt = pt.copy(); tt[:, :, :2] = 0.0
    lab = np.zeros((10, 4), dtype=np.int8)
    vt = ep.compute_valid_target(tt)
    res = cf._eval_groups(pt, tt, 20.0, lab, vt, 0.9)
    check('n_valid=240', res['n_valid'] == 240)
    check('cov=1 on valid', res['overall_cov'] == 1.0)


def test_quantile_valid_only():
    """uncond quantile uses only valid residuals."""
    pv = np.full((10, 12, 4), 50.0)
    tv = pv.copy(); tv[:, :, :2] = 0.0
    pt = pv; tt = tv
    lab = np.zeros((10, 4), dtype=np.int8)
    vv = ep.compute_valid_target(tv); vt = ep.compute_valid_target(tt)
    res, q = cf.unconditional(pv, tv, pt, tt, lab, 0.1, valid_val=vv, valid_test=vt)
    check('q=0 (valid residual all 0)', q == 0.0)


if __name__ == '__main__':
    print('[test_valid_mask]')
    test_mask_conservation()
    test_target_mask()
    test_quantile_valid_only()
    print(f'ALL {len(PASS)} PASS')
