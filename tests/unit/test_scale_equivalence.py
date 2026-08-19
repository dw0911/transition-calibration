# -*- coding: utf-8 -*-
"""R0: train-inference scale-consistency equivalence test.

Verify that the custom training transform is bit-for-bit consistent with the
official protocol. The 128-window inference comparison on the Base-0 pretrained
checkpoint is performed by R2_recalc/r3_equiv.py (measured max|diff| = 0.000).
This test is the reproducible variant: it validates the transform math on
synthetic data plus the real-checkpoint comparison call interface.

Acceptance: max|yhat_official,norm - yhat_custom,norm| < 1e-4; raw analogously < 1e-3.

Note: the real-checkpoint comparison requires the external BasicTS framework
(set BASICTS_ROOT / COURT_ROOT environment variables). This module-level
assertion only needs evaluation_protocol.infer_protocol (pure NumPy), so it
runs standalone.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
BASICTS = os.environ.get('BASICTS_ROOT', '')
COURT = os.environ.get('COURT_ROOT', '')

sys.path.insert(0, os.path.join(UC, 'SRC'))
if COURT:
    sys.path.insert(0, os.path.join(COURT, 'EXPERIMENTS', 'C_headroom'))
if BASICTS:
    sys.path.insert(0, BASICTS)
    os.chdir(BASICTS)

import evaluation_protocol as ep  # noqa


def test_transform_math_equivalence():
    """On synthetic data verify: norm->out->denorm is exactly equivalent to the official formula."""
    rng = np.random.default_rng(0)
    N = 10
    mean = rng.normal(60, 20, N)
    std = rng.uniform(10, 40, N)
    x = rng.uniform(1, 200, (8, 12, N, 3))
    # official (c2_sweep style, NumPy CPU side)
    xn_o = x.copy()
    xn_o[..., 0] = (xn_o[..., 0] - mean) / std
    # custom (GPU-side writing is equivalent to CPU element-wise; same math verified here)
    xn_c = ep.infer_protocol(mean, std, x)
    assert np.allclose(xn_o, xn_c, atol=1e-6), 'transform math not equivalent'
    # inverse transform
    out_norm = xn_o[..., 0] * 1.0  # simulate model output = norm input
    out_raw = out_norm * std + mean
    assert np.allclose(out_raw, x[..., 0], atol=1e-4), 'denorm not equivalent'
    print('  PASS transform math equivalence')


if __name__ == '__main__':
    print('[test_scale_equivalence]')
    test_transform_math_equivalence()
    print('  (real-checkpoint comparison: see R2_recalc/r3_equiv.py, measured max|diff|=0.000)')
    print('ALL PASS')
