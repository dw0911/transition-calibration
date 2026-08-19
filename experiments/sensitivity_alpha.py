# -*- coding: utf-8 -*-
"""Alpha sensitivity experiment (Tier-2, T2.4).

For PeMS04 STAEformer (seed 42), sweep alpha in {0.05, 0.10, 0.20} and compare
unconditional vs severity-conditioned Mondrian (K=5) on per-group coverage and WCE.

Output: experiments/artifacts/sensitivity_alpha_PEMS04.json
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

K = 5
IL, OL = 12, 12
DATASET = 'PEMS04'
N = 307
SEED = 42
CKPT = os.path.join(os.environ['REPRO_CKPT_DIR'],
                    f'{DATASET}_r3_4split_seed{SEED}', 'best.pt')
ALPHA_LIST = [0.05, 0.10, 0.20]


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

    tau = R.tx.fit_per_node_q95(full[:train_end, :, 0])
    stats = R.sev.fit_severity_stats(full[:train_end, :, 0])
    s_series = R.sev.severity_series(full[:, :, 0], stats)

    g_test = np.arange(calib_end, T - IL - OL + 1)
    labels = R.tx.classify_from_flow(full[:, :, 0], g_test, tau, IL, OL)
    r_test = R.sev.window_in_severity(s_series, g_test, IL)

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
    r = r_test[:n]
    vv, vt = (tt > 0), (tt > 0)

    out = {'date': '2026-08-19', 'protocol': 'sensitivity_alpha_PEMS04', 'K': K}
    for alpha in ALPHA_LIST:
        res_u, q = R.cf.unconditional(pt, tt, pt, tt, lab, alpha,
                                      valid_val=vv, valid_test=vt)
        res_s, _ = R.cf.mondrian_hard(pt, tt, pt, tt, lab, r, r, K=K,
                                       alpha=alpha, valid_val=vv, valid_test=vt)
        out[f'alpha={alpha}'] = {
            'unconditional': summarize(res_u),
            'severity': summarize(res_s),
        }
        print(f'alpha={alpha}: uncond WCE={res_u["WCE"]:.4f} '
              f'sev WCE={res_s["WCE"]:.4f} '
              f'uncond TypeC={res_u["TypeC"]["cov"]:.3f} '
              f'sev TypeC={res_s["TypeC"]["cov"]:.3f}')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'artifacts', 'sensitivity_alpha_PEMS04.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()