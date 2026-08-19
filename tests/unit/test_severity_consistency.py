# -*- coding: utf-8 -*-
"""R1: severity cross-path consistency (P0-4 acceptance).

STAEformer (full-series window_in_severity) and STID/legacy (windowed
window_in_severity_from_windows) must return exactly the same r^in for the same
raw sequence and same starts: max|r_STAE - r_STID| < 1e-6.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
import calibration.severity as sev  # noqa

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


def test_cross_path_equality():
    rng = np.random.default_rng(1)
    T, N = 500, 6
    flow = 100 + np.cumsum(rng.normal(0, 5, (T, N)), axis=0).clip(min=1)
    flow[::50, 1] = 0.0  # zero out to test mask
    stats = sev.fit_severity_stats(flow[:300])
    s = sev.severity_series(flow, stats)
    gs = np.arange(300, 400)
    # series version
    r_seq = sev.window_in_severity(s, gs, 12)
    # windowed version
    x_win = np.stack([flow[i:i + 12] for i in gs])
    r_win = sev.window_in_severity_from_windows(x_win, stats)
    d = float(np.abs(r_seq - r_win).max())
    check(f'cross-path r^in max|diff|={d:.2e} < 1e-6', d < 1e-6)
    # Q99 clip active
    check('s <= q99', float(s.max()) <= stats.q99 + 1e-12)


if __name__ == '__main__':
    print('[test_severity_consistency]')
    test_cross_path_equality()
    print(f'ALL {len(PASS)} PASS')
