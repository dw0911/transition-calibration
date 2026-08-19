# -*- coding: utf-8 -*-
"""Benchmark the runtime overhead of the severity-conditioned calibration layer.

Measures (on CPU, single thread):
  1. severity_series over a full PeMS04-length series (T=16992, N=307)
  2. window_in_severity over the test windows (n_windows ~= 3200, N=307)
  3. per-window severity cost (i.e., 1+2 amortized over n_windows)

Reported as absolute ms and as a fraction of a single 12-step forecast batch.
"""
import os
import sys
import time

import numpy as np

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPRO_ROOT)
sys.path.insert(0, os.path.join(REPRO_ROOT, 'SRC'))

import calibration.severity as sev  # noqa: E402

T, N = 16992, 307          # PeMS04 dimensions
L, H = 12, 12
n_win = 3200


def bench(fn, *args, repeat=5):
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    rng = np.random.default_rng(0)
    flow = np.maximum(100 + np.cumsum(rng.normal(0, 5, (T, N)), axis=0), 1.0)

    # train statistics fit (once, offline)
    ms_fit = bench(sev.fit_severity_stats, flow[:int(0.6 * T)])
    stats = sev.fit_severity_stats(flow[:int(0.6 * T)])

    # severity series (full series, once per forecast cycle is unnecessary;
    # only the window suffix is needed, but we report the full-series cost)
    ms_series = bench(sev.severity_series, flow, stats)
    s = sev.severity_series(flow, stats)

    # window in-severity (per test window set)
    gs = np.arange(int(0.8 * T), int(0.8 * T) + n_win)
    ms_win = bench(sev.window_in_severity, s, gs, L)

    # per-window amortized cost (series cost shared over all windows of a cycle)
    per_win = (ms_series + ms_win) / n_win
    print(f'fit_severity_stats (once):  {ms_fit:8.2f} ms')
    print(f'severity_series (T={T}):    {ms_series:8.2f} ms')
    print(f'window_in_severity (n={n_win}): {ms_win:8.2f} ms')
    print(f'per-window severity cost:   {per_win:8.4f} ms')
    print(f'-> per-window overhead vs a single STAEformer test batch '
          f'(~1-3 ms on GPU): ~{per_win:8.4f} ms (CPU, single thread)')

    out = {
        'fit_ms': ms_fit, 'series_ms': ms_series, 'window_ms': ms_win,
        'per_window_ms': per_win, 'n_windows': n_win,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'artifacts', 'benchmark_severity.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved -> {path}')


if __name__ == '__main__':
    main()
