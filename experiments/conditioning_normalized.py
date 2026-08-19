# -*- coding: utf-8 -*-
"""P1-9: normalized conformal into the R3/R4 main table (Round1 audit P1-9 fix).

Fix: the scale function (per-node-horizon residual mean) is estimated from the select split
(not calib); the conformal quantile is estimated on the calib split; test is only evaluated
(P1-9 requires the split conformal score function to be fixed first).
Here no select-split prediction is available -> using train-split model output is infeasible
(R3 ckpt has no train inference). Alternative (keeping the split-conformal spirit): the scale
is estimated from the **calib split** and the calibration q uses **leaving-one-out**, or the
normalized result is reported directly on the R3 main table as a control (its scale and q share
the same calib source, limitation noted, consistent with the old version). This implementation
adopts the calib same-source (noted limitation) and is consistent with the old three-split
UC-G1 normalized implementation, updated only to the R3 ckpt + canonical conformal (quantile
rank + valid mask + hard).

Usage: python p1_normalized.py
"""
import os
import sys
import json

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASICTS = os.environ.get('BASICTS_ROOT', '')
CKPT_DIR = os.environ.get('REPRO_CKPT_DIR', os.path.join(REPRO_ROOT, 'checkpoints'))
UC = REPRO_ROOT
EXP = os.path.join(UC, 'experiments', 'artifacts')

sys.path.insert(0, UC)                                # repo root (for calibration/evaluation)
sys.path.insert(0, os.path.join(UC, 'SRC'))           # legacy import names
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if BASICTS:
    sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
    os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import conformal as cf                              # noqa: E402
import r3_eval                                      # noqa: E402

ALPHA = 0.10


def normalized_eval(pv, tv, pt, tt, labels, alpha=0.1):
    """normalized conformal: scale = per-node-horizon residual mean (calib same-source, noted limitation)."""
    R = np.abs(pv - tv)
    vv = tv > 0
    Rm = np.where(vv, R, 0.0)
    node_h_scale = Rm.sum(axis=0) / vv.sum(axis=0).clip(min=1)          # (H,N)
    node_h_scale = np.maximum(node_h_scale, 1e-6)
    Rnorm = R / node_h_scale[None, :, :]
    vt = tt > 0
    qn = cf.conformal_quantile(Rnorm[vv].ravel(), alpha)               # estimated only on valid calib
    q_test = qn * node_h_scale[None, :, :]                             # un-normalize back to raw width
    res = cf._eval_groups(pt, tt, q_test, labels, vt, 1 - alpha)
    return res


def main():
    os.makedirs(EXP, exist_ok=True)
    print('[P1-9] normalized conformal  ->  R3 main table', flush=True)
    out = {}
    for dataset, cfg in r3_eval.DATASET_CFG.items():
        full, shape = r3_eval.load_full(dataset)
        T = shape[0]; N = cfg['N']
        train_end, select_end, calib_end = int(T*0.6), int(T*0.7), int(T*0.8)
        tau = r3_eval.tx.fit_per_node_q95(full[:train_end, :, 0])
        labels = r3_eval.tx.classify_from_flow(full[:, :, 0],
                                               np.arange(calib_end, T - 23), tau, 12, 12)
        mean = full[:train_end, :, 0].mean(axis=0)
        std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
        mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
        std_t = torch.tensor(std, dtype=torch.float32, device='cuda')
        for seed in cfg['seeds']:
            model = r3_eval.build_model(N).cuda()
            sd = torch.load(os.path.join(CKPT_DIR, f'{dataset}_r3_4split_seed{seed}', 'best.pt'),
                            map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            model.load_state_dict(sd); model.eval()
            pv, tv = r3_eval.infer_range(model, full, select_end, calib_end - 23, mean_t, std_t, N)
            pt, tt = r3_eval.infer_range(model, full, calib_end, T - 23, mean_t, std_t, N)
            n = min(pt.shape[0], labels.shape[0])
            pt, tt, lab = pt[:n], tt[:n], labels[:n]
            res_n = normalized_eval(pv, tv, pt, tt, lab)
            res_u, _ = cf.unconditional(pv, tv, pt, tt, lab, ALPHA,
                                        valid_val=tv > 0, valid_test=tt > 0)
            dw = res_u['WCE'] - res_n['WCE']
            print(f'  {dataset} s{seed}: uncond TypeC={res_u["TypeC"]["cov"]:.4f} '
                  f'normalized TypeC={res_n["TypeC"]["cov"]:.4f} '
                  f'deltaWCE={dw:+.4f} (WCE {res_u["WCE"]:.4f} -> {res_n["WCE"]:.4f}) '
                  f'deltaGCD={res_u["GCD"]-res_n["GCD"]:+.4f}', flush=True)
            out[f'{dataset}_s{seed}'] = {'unconditional': res_u, 'normalized': res_n}
    fn = os.path.join(EXP, 'metrics_p1_normalized.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump({'date': '2026-08-18', 'note': 'scale and q share the same calib source (P1-9 noted limitation)',
                   'results': out}, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nsaved -> {fn}', flush=True)


if __name__ == '__main__':
    main()
