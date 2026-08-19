# -*- coding: utf-8 -*-
"""Mine a real TypeC transition window on PeMS04 for the deployment case study.

Reuses the reproduce_table4 pipeline: load PeMS04 data, build STAEformer (seed 42),
infer the test split, compute taxonomy labels and severity, then select a representative
TypeC window (output transition with high severity and a large coverage miss) and dump
the per-step series needed for the case-study figure.

Output: case_study_window.json (flow, pred, true, severity, unconditional interval,
severity-calibrated interval, labels).
"""
import os
import sys
import json

import numpy as np

# ---- paths ----
REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPRO_ROOT, 'experiments'))

# Set fallback paths here (only used when the environment variables are unset).
if not os.environ.get('BASICTS_ROOT'):
    os.environ['BASICTS_ROOT'] = r'f:\个人\HF\交通预测\BasicTS058'
if not os.environ.get('REPRO_CKPT_DIR'):
    os.environ['REPRO_CKPT_DIR'] = r'f:\个人\HF\交通预测\uncalib_v1\EXPERIMENTS\UC-G3\checkpoints'

BASICTS = os.environ.get('BASICTS_ROOT', '')
CKPT = os.path.join(os.environ.get('REPRO_CKPT_DIR', ''),
                    'PEMS04_r3_4split_seed42', 'best.pt')

import torch
import numpy as _np  # noqa

import reproduce_table4 as R  # noqa: E402

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
IL, OL = 12, 12
DATASET = 'PEMS04'
N = 307


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
    std = full[:train_end, :, 0].std(axis=0)
    std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

    model = R.build_model(N).cuda()
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()

    pt, tt = R.infer_range(model, full, calib_end, T - IL - OL + 1, mean_t, std_t, N)
    n = min(pt.shape[0], labels.shape[0])
    pt, tt, lab = pt[:n], tt[:n], labels[:n]
    r = r_test[:n]
    vv, vt = (tt > 0), (tt > 0)

    # calibration: unconditional q and severity-conditioned Mondrian
    res_u, q = R.cf.unconditional(pt, tt, pt, tt, lab, ALPHA,
                                  valid_val=vv, valid_test=vt)
    res_s, meta_s = R.cf.mondrian_hard(pt, tt, pt, tt, lab, r, r, K=5,
                                       alpha=ALPHA, valid_val=vv, valid_test=vt)
    edges = np.array(meta_s['edges'])
    qb = np.array(meta_s['q_bins'])
    b = np.clip(np.digitize(np.broadcast_to(r[:, None, :], pt.shape), edges[1:-1]), 0, 4)
    q_test = qb[b]  # (n,12,N) per-window per-node q

    # pick a representative window: TypeC, large residual, big coverage miss
    covered_u = (tt >= pt - q) & (tt <= pt + q)
    covered_s = (tt >= pt - q_test) & (tt <= pt + q_test)
    type_c = (lab == 3)
    miss_u = (~covered_u) & type_c[:, None, :]
    resid = np.abs(pt - tt)

    candidates = []
    for w in range(n):
        # a node with a clear TypeC output transition and a missed unconditional interval
        nodes = np.where(type_c[w])[0]
        if len(nodes) == 0:
            continue
        for nd in nodes:
            if not np.any(miss_u[w, :, nd]):
                continue
            if tt[w, :, nd].max() < 40:  # skip near-zero traffic
                continue
            score = (float(resid[w, :, nd].mean())
                     + float(r[w, nd]) * 10.0)
            candidates.append((score, w, nd))

    candidates.sort(reverse=True)
    print(f'candidate TypeC windows: {len(candidates)}')
    for k in range(min(8, len(candidates))):
        sc, w, nd = candidates[k]
        g = int(g_test[w])
        print(f'  cand#{k}: window={w} g={g} node={nd} '
              f'severity={float(r[w, nd]):.2f} '
              f'resid_mean={float(resid[w, :, nd].mean()):.2f} '
              f'flow_in=[{float(full[g, nd, 0]):.0f}..{float(full[g+IL-1, nd, 0]):.0f}] '
              f'flow_out=[{float(full[g+IL, nd, 0]):.0f}..{float(full[g+IL+OL-1, nd, 0]):.0f}]')

    # export the best candidate
    if candidates:
        sc, w, nd = candidates[0]
        g = int(g_test[w])
        series_flow = full[g - 6: g + IL + OL + 6, nd, 0].astype(float).tolist()
        out = {
            'window_idx': int(w), 'start_g': g, 'node': int(nd),
            'severity': float(r[w, nd]),
            'q_uncond': float(q),
            'q_sev': float(q_test[w, :, nd].max()),
            'flow_series': series_flow,
            'pred': pt[w, :, nd].astype(float).tolist(),
            'true': tt[w, :, nd].astype(float).tolist(),
            'severity_series': s_series[g - 6: g + IL + OL + 6, nd].astype(float).tolist(),
            'label': int(lab[w, nd]),
            'input_flow': full[g:g + IL, nd, 0].astype(float).tolist(),
            'output_flow': full[g + IL:g + IL + OL, nd, 0].astype(float).tolist(),
        }
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'case_study_window.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f'\nexported case_study_window.json: g={g} node={nd} label={int(lab[w, nd])}')
    else:
        print('no candidate found')


if __name__ == '__main__':
    main()
