# -*- coding: utf-8 -*-
"""Severity-statistic sensitivity experiment (Tier-2, T2.3).

Three robust/standard statistics for the per-node severity:
  - median + 1.4826*MAD   (our default; robust)
  - mean   + std          (standard)
  - 75th-percentile + (75-25th) percentile range  (simple, robust)

Output: experiments/artifacts/sensitivity_stats_PEMS04.json
"""
import os
import sys
import json

import numpy as np

if not os.environ.get('BASICTS_ROOT'):
    os.environ['BASICTS_ROOT'] = r'f:\个人\HF\交通预测\BasicTS058'
if not os.environ.get('REPRO_CKPT_DIR'):
    os.environ['REPRO_CKPT_DIR'] = r'f:\个人\HF\交通预测\uncalib_v1\EXPERIMENTS\UC-G3\checkpoints'

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPRO_ROOT)
sys.path.insert(0, os.path.join(REPRO_ROOT, 'experiments'))
sys.path.insert(0, os.path.join(REPRO_ROOT, 'SRC'))

import torch  # noqa: E402
import reproduce_table4 as R  # noqa: E402

ALPHA = 0.10
IL, OL = 12, 12
K = 5
DATASET = 'PEMS04'
N = 307
SEED = 42
CKPT = os.path.join(os.environ['REPRO_CKPT_DIR'],
                    f'{DATASET}_r3_4split_seed{SEED}', 'best.pt')


def fit_stats_alt(flow_train, mode):
    """Alternative per-node severity statistics; returns SeverityStats-like object.
    Returns (m, d, q99). Uses only valid diffs (zero-masked).
    """
    x = np.asarray(flow_train, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=0))
    ok = (x[1:] > 0) & (x[:-1] > 0)
    _, N = ad.shape
    m = np.zeros(N); d = np.ones(N)
    for n in range(N):
        v = ad[ok[:, n], n]
        if v.size < 10:
            v = ad[ok]
        if mode == 'median_mad':
            med = np.median(v); mad = np.median(np.abs(v - med))
            m[n] = med; d[n] = 1.4826 * mad if mad > 0 else max(np.std(v), 1e-6)
        elif mode == 'mean_std':
            m[n] = float(v.mean()); d[n] = float(v.std()) if v.std() > 0 else 1.0
        elif mode == 'percentile':
            p25 = float(np.percentile(v, 25)); p75 = float(np.percentile(v, 75))
            m[n] = p25; d[n] = (p75 - p25) if (p75 - p25) > 0 else max(np.std(v), 1e-6)
    s_train = np.where(ok, np.abs(ad - m) / d, 0.0)
    valid_s = s_train[ok]
    q99 = float(np.quantile(valid_s, 0.99)) if valid_s.size > 0 else float('inf')
    return m, d, q99


def severity_series_alt(flow, m, d, q99):
    x = np.asarray(flow, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=0))
    ok = (x[1:] > 0) & (x[:-1] > 0)
    s_raw = np.abs(ad - m) / d
    s = np.where(ok, np.minimum(s_raw, q99), 0.0)
    return s


def summarize(res):
    return {
        'Normal': float(res['Normal']['cov']),
        'TypeA':  float(res['TypeA']['cov']),
        'TypeB':  float(res['TypeB']['cov']),
        'TypeC':  float(res['TypeC']['cov']),
        'WCE':    float(res['WCE']),
        'GCD':    float(res['GCD']),
        'width':  float(res['mean_width']),
    }


def main():
    full, shape = R.load_full(DATASET)
    T = shape[0]
    train_end, select_end, calib_end = int(T * 0.6), int(T * 0.7), int(T * 0.8)

    labels = R.tx.classify_from_flow(full[:, :, 0],
                                     np.arange(calib_end, T - IL - OL + 1),
                                     R.tx.fit_per_node_q95(full[:train_end, :, 0]),
                                     IL, OL)

    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

    model = R.build_model(N).cuda()
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd); model.eval()

    pt, tt = R.infer_range(model, full, calib_end, T - IL - OL + 1, mean_t, std_t, N)
    n = min(pt.shape[0], labels.shape[0])
    pt, tt, lab = pt[:n], tt[:n], labels[:n]
    vv, vt = (tt > 0), (tt > 0)

    out = {'date': '2026-08-19', 'protocol': 'sensitivity_stats_PEMS04', 'K': K}
    for mode in ['median_mad', 'mean_std', 'percentile']:
        m, d, q99 = fit_stats_alt(full[:train_end, :, 0], mode)
        s = severity_series_alt(full[:, :, 0], m, d, q99)
        g_test = np.arange(calib_end, T - IL - OL + 1)
        r = R.sev.window_in_severity(s, g_test, IL)[:n]
        res_s, _ = R.cf.mondrian_hard(pt, tt, pt, tt, lab, r, r, K=K,
                                      alpha=ALPHA, valid_val=vv, valid_test=vt)
        out[mode] = summarize(res_s)
        print(f'{mode}: Normal={res_s["Normal"]["cov"]:.3f} '
              f'TypeA={res_s["TypeA"]["cov"]:.3f} TypeB={res_s["TypeB"]["cov"]:.3f} '
              f'TypeC={res_s["TypeC"]["cov"]:.3f} '
              f'WCE={res_s["WCE"]:.4f} GCD={res_s["GCD"]:.4f} '
              f'width={res_s["mean_width"]:.2f}')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'artifacts', 'sensitivity_stats_PEMS04.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()