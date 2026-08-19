# -*- coding: utf-8 -*-
"""R4: window-level paired stationary bootstrap method-effect CI (P0-7 correct implementation).

Principle (audit P0-7):
  - bootstrap unit = forecast window (temporal structure preserved)
  - per replicate: stationary bootstrap resamples windows -> pooled numerator/denominator,
    then recomputes each method's group coverage -> WCE/GCD -> deltaWCE = WCE_uncond - WCE_method
  - not iid sampling flat elements; no downsampling that loses time blocks
  - block-length sensitivity: p in {1/144, 1/288, 1/576} (mean block 0.5/1/2 days)
  - 3-seed summary: sample SD ddof=1 (P1-11)

Usage: python r4_bootstrap.py
"""
import os
import sys
import json

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASICTS = os.environ.get('BASICTS_ROOT', '')
UC = REPRO_ROOT
EXP = os.path.join(UC, 'experiments', 'artifacts')

sys.path.insert(0, os.path.join(UC, 'SRC'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if BASICTS:
    sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
    os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import conformal as cf                              # noqa: E402
import r3_eval                                      # noqa: E402

ALPHA = 0.10
NOMINAL = 1 - ALPHA
N_GROUP = {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}


def per_window_stats(pred, true, q, labels):
    """q is scalar or (n,H,N). Returns per-window covered num/valid den per group (n,4)."""
    n = pred.shape[0]
    covered = (true >= pred - q) & (true <= pred + q) & (true > 0)
    num = np.zeros((n, 4), dtype=np.float64)
    den = np.zeros((n, 4), dtype=np.float64)
    for tid in range(4):
        m = np.broadcast_to((labels == tid)[:, None, :], covered.shape) & (true > 0)
        num[:, tid] = (covered & m).sum(axis=(1, 2))
        den[:, tid] = m.sum(axis=(1, 2))
    return num, den


def wce_gcd_from_pooled(num, den, nominal=NOMINAL):
    """pooled counts  ->  WCE/GCD + responsible group. num/den: (4,)"""
    covs = np.zeros(4)
    for tid in range(4):
        covs[tid] = num[tid] / den[tid] if den[tid] > 0 else np.nan
    wce = float(np.nanmax(np.abs(covs - nominal)))
    gcd = float(np.nanmax(covs) - np.nanmin(covs))
    gaps = {N_GROUP[t]: abs(covs[t] - nominal) for t in range(4) if not np.isnan(covs[t])}
    wg = max(gaps, key=gaps.get)
    return wce, gcd, wg


def method_q(pred_val, true_val, pred_test, true_test, labels, cond, q_uncond, mode='hard'):
    """Compute each method's q array on test. cond can be None (uncond)."""
    if cond is None:
        return np.full(pred_test.shape, q_uncond)
    cvc, cvt = cond
    R = np.abs(pred_val - true_val)
    vv = true_val > 0
    if mode == 'hard':
        _, meta = cf.mondrian_hard(pred_val, true_val, pred_test, true_test, labels,
                                   cvc, cvt, K=5, alpha=ALPHA, valid_val=vv,
                                   valid_test=true_test > 0)
    else:
        _, meta = cf.mondrian_interpolated(pred_val, true_val, pred_test, true_test, labels,
                                           cvc, cvt, K=5, alpha=ALPHA, valid_val=vv,
                                           valid_test=true_test > 0)
    edges = np.array(meta['edges']); qb = np.array(meta['q_bins'])
    ct = np.broadcast_to(np.asarray(cvt)[:, None, :], pred_test.shape)
    b = np.clip(np.digitize(ct, edges[1:-1]), 0, 4)
    if mode == 'hard':
        return qb[b]
    # interp consistent with conformal._mondrian
    anchors = np.array(meta['anchors'])
    return np.interp(ct, anchors, qb)


def stationary_bootstrap_diff(numU, denU, numM, denM, n_boot=1000, seed=42, p=1.0/288):
    """stationary bootstrap CI for deltaWCE / deltaGCD (resampling the same windows for both methods' pooled counts)."""
    rng = np.random.default_rng(seed)
    n_win = numU.shape[0]
    dw, dg = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n_win, dtype=np.int64)
        idx[0] = rng.integers(n_win)
        for i in range(1, n_win):
            idx[i] = rng.integers(n_win) if rng.random() < p else (idx[i - 1] + 1) % n_win
        nu, du = numU[idx].sum(axis=0), denU[idx].sum(axis=0)
        nm, dm = numM[idx].sum(axis=0), denM[idx].sum(axis=0)
        wu, gu, _ = wce_gcd_from_pooled(nu, du)
        wm, gm, _ = wce_gcd_from_pooled(nm, dm)
        dw[b] = wu - wm
        dg[b] = gu - gm
    return (float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5)), float(dw.mean())), \
           (float(np.percentile(dg, 2.5)), float(np.percentile(dg, 97.5)), float(dg.mean()))


def main():
    os.makedirs(EXP, exist_ok=True)
    print('[R4] window-level paired stationary bootstrap', flush=True)
    P_LIST = [1.0/144, 1.0/288, 1.0/576]
    summary = {}
    for dataset, cfg in r3_eval.DATASET_CFG.items():
        full, shape = r3_eval.load_full(dataset)
        T = shape[0]; N = cfg['N']
        train_end, select_end, calib_end = int(T*0.6), int(T*0.7), int(T*0.8)
        tau = r3_eval.tx.fit_per_node_q95(full[:train_end, :, 0])
        stats = r3_eval.sev.fit_severity_stats(full[:train_end, :, 0])
        s_series = r3_eval.sev.severity_series(full[:, :, 0], stats)
        g_calib = np.arange(select_end, calib_end - 24 + 1)
        g_test = np.arange(calib_end, T - 24 + 1)
        r_c = r3_eval.sev.window_in_severity(s_series, g_calib, 12)
        r_t = r3_eval.sev.window_in_severity(s_series, g_test, 12)
        f_c = full[g_calib[:, None] + np.arange(12)][:, :, :, 0].mean(axis=1)
        f_t = full[g_test[:, None] + np.arange(12)][:, :, :, 0].mean(axis=1)
        labels = r3_eval.tx.classify_from_flow(full[:, :, 0], g_test, tau, 12, 12)
        mean = full[:train_end, :, 0].mean(axis=0)
        std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
        mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
        std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

        for seed in cfg['seeds']:
            print(f'\n=== {dataset} s{seed} ===', flush=True)
            model = r3_eval.build_model(N).cuda()
            sd = torch.load(os.path.join(UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints',
                                         f'{dataset}_r3_4split_seed{seed}', 'best.pt'),
                            map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            model.load_state_dict(sd); model.eval()
            pv, tv = r3_eval.infer_range(model, full, select_end, calib_end - 23,
                                         mean_t, std_t, N)
            pt, tt = r3_eval.infer_range(model, full, calib_end, T - 23, mean_t, std_t, N)
            n = min(pt.shape[0], labels.shape[0])
            pt, tt, lab = pt[:n], tt[:n], labels[:n]
            rc, rt = r_c[:pv.shape[0]], r_t[:n]
            fc, ft = f_c[:pv.shape[0]], f_t[:n]
            pmc, pmt = np.abs(pv).mean(axis=1), np.abs(pt).mean(axis=1)
            vv, vt = tv > 0, tt > 0

            res_u, q = cf.unconditional(pv, tv, pt, tt, lab, ALPHA, valid_val=vv, valid_test=vt)
            numU, denU = per_window_stats(pt, tt, q, lab)
            conds = {'severity': (rc, rt), 'flow_level': (fc, ft), 'predicted_mag': (pmc, pmt)}
            seeds_out = {}
            for cname, cond in conds.items():
                qM = method_q(pv, tv, pt, tt, lab, cond, q, 'hard')
                numM, denM = per_window_stats(pt, tt, qM, lab)
                srow = {}
                for plabel, p in zip(['block144', 'block288', 'block576'], P_LIST):
                    (dl, dh, dm), (gl, gh, gm) = stationary_bootstrap_diff(
                        numU, denU, numM, denM, n_boot=1000, seed=42, p=p)
                    srow[plabel] = {'dWCE': {'lo': dl, 'hi': dh, 'mean': dm},
                                    'dGCD': {'lo': gl, 'hi': gh, 'mean': gm}}
                seeds_out[cname] = srow
                d = srow['block288']
                ok_w = d['dWCE']['lo'] > 0; ok_g = d['dGCD']['lo'] > 0
                print(f"  [{cname}] deltaWCE={d['dWCE']['mean']:+.4f} "
                      f"CI=[{d['dWCE']['lo']:+.4f},{d['dWCE']['hi']:+.4f}] "
                      f"{'✓' if ok_w else '✗'}  deltaGCD={d['dGCD']['mean']:+.4f} "
                      f"CI=[{d['dGCD']['lo']:+.4f},{d['dGCD']['hi']:+.4f}] "
                      f"{'✓' if ok_g else '✗'} (block=288)", flush=True)
            summary[f'{dataset}_s{seed}'] = seeds_out

    # 3-seed ddof=1 summary (P1-11)
    print('\n=== 3-seed summary (block288, ddof=1)===', flush=True)
    final = {}
    for cname in ['severity', 'flow_level', 'predicted_mag']:
        dws = np.array([summary[f'PEMS04_s{s}'][cname]['block288']['dWCE']['mean']
                        for s in [42, 123, 456]])
        dgs = np.array([summary[f'PEMS04_s{s}'][cname]['block288']['dGCD']['mean']
                        for s in [42, 123, 456]])
        final[cname] = {'deltaWCE_mean': float(dws.mean()), 'deltaWCE_std_ddof1': float(dws.std(ddof=1)),
                        'deltaWCE_all_pos': bool((dws > 0).all()),
                        'deltaGCD_mean': float(dgs.mean()), 'deltaGCD_std_ddof1': float(dgs.std(ddof=1)),
                        'deltaGCD_all_pos': bool((dgs > 0).all())}
        print(f"  {cname:14s} deltaWCE={dws.mean():+.4f}+/-{dws.std(ddof=1):.4f} "
              f"3/3>0:{(dws > 0).all()}  deltaGCD={dgs.mean():+.4f}+/-{dgs.std(ddof=1):.4f} "
              f"3/3>0:{(dgs > 0).all()}", flush=True)
    out = {'date': '2026-08-18', 'protocol': 'R4 window-level paired stationary bootstrap',
           'block_lengths': {'144': '0.5day', '288': '1day', '576': '2day'},
           'per_config': summary, 'pems04_3seed_summary_ddof1': final}
    fn = os.path.join(EXP, 'metrics_r4_bootstrap.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nsaved -> {fn}', flush=True)


if __name__ == '__main__':
    main()
