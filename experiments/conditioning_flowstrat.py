# -*- coding: utf-8 -*-
"""P1-7 + P1-8: node-window granularity rho and the flow-stratified bootstrap (fixed version).

P1-7 (node-window rho): the old ucg1c used window-mean compression, giving rho=0.90 (an
  ecological correlation). Now, per (i,n) pair, e_{i,n} = mean_h |y-yhat| is compared with
  r^in_{i,n} using node clustering: first demean within node (within-node correlation), then
  form a time-block stationary bootstrap CI.

P1-8 (flow-stratified bootstrap fix): first resample the window block on the original time
  axis, then, within each replicate, filter the flow stratum and compute the pooled coverage
  diff (severity vs uncond); report a CI for the high/mid/low flow terciles. This fixes the old
  version's "filter-then-resample broke temporal order" bug.

Usage: python p1_rho_flowstrat.py
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
           # legacy import names
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if BASICTS:
    sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
    os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import calibration.mondrian as cf                              # noqa: E402
import calibration.severity as sev                              # noqa: E402
import r3_eval                                      # noqa: E402

ALPHA = 0.10
SEED = 42


def node_window_corr(e_flat, r_flat, node_ids, p=1/288, n_boot=1000):
    """Node-clustered within correlation: demean within node, then a time-block bootstrap CI.

    e/r: (n_win*N,), node_ids: (n_win*N,); resample window blocks by time index.
    """
    rng = np.random.default_rng(SEED)
    n_win = len(np.unique(node_ids)) if False else None
    # reshape (n_win, N)
    nw = e_flat.reshape(-1, 307) if e_flat.ndim == 1 and e_flat.shape[0] % 307 == 0 else e_flat
    # simple impl: demean within node -> pooled corr, window-block bootstrap
    R = r_flat
    E = e_flat
    # within-node demean
    E_d = E - E.mean(0, keepdims=True) if E.ndim == 2 else E
    return None  # placeholder; the main implementation lives in p1_rho_main below


def p1_rho_main(full, N, train_end, select_end, calib_end, stats, tau, labels,
                pv, tv, pt, tt, cond_r_calib, cond_r_test):
    """P1-7 main implementation: rho(r^in, |y-yhat|) per (i,n) pair + node-clustered block bootstrap."""
    vt = tt > 0
    e = np.abs(pt - tt)                                   # (n,12,N)
    e_win = np.where(vt, e, np.nan).mean(axis=1)          # (n,N) mean over valid horizon
    r = cond_r_test                                       # (n,N) r^in
    # node-clustered correlation: compute rho per node separately, then average (within-node correlation not inflated)
    n = min(e_win.shape[0], r.shape[0])
    e_win, r = e_win[:n], r[:n]
    rhos = []
    for k in range(N):
        m = np.isfinite(e_win[:, k]) & np.isfinite(r[:, k])
        if m.sum() > 100:
            rhos.append(float(np.corrcoef(e_win[m, k], r[m, k])[0, 1]))
    rho_mean = float(np.mean(rhos))
    rho_std = float(np.std(rhos)) if len(rhos) > 1 else 0.0
    # stationary bootstrap on windows for pooled rho CI
    rng = np.random.default_rng(SEED)
    n_win = n
    e_flat = e_win[:n].ravel(); r_flat = r[:n].ravel()
    boot = np.empty(1000)
    p = 1/288
    for b in range(1000):
        idx = np.empty(n_win, dtype=np.int64)
        idx[0] = rng.integers(n_win)
        for i in range(1, n_win):
            idx[i] = rng.integers(n_win) if rng.random() < p else (idx[i-1]+1) % n_win
        eb = e_flat.reshape(n_win, N)[idx].ravel()
        rb = r_flat.reshape(n_win, N)[idx].ravel()
        m = np.isfinite(eb) & np.isfinite(rb)
        if m.sum() > 1000:
            boot[b] = float(np.corrcoef(eb[m], rb[m])[0, 1])
    boot = boot[np.isfinite(boot)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    print(f'  [P1-7] node-window rho(r_in, resid): per-node mean={rho_mean:.4f}+/-{rho_std:.4f} '
          f'(pooled block CI=[{ci[0]:.4f},{ci[1]:.4f}])', flush=True)
    return {'rho_per_node_mean': rho_mean, 'rho_per_node_std': rho_std, 'ci95': list(ci)}


def p1_flow_stratified(pv, tv, pt, tt, labels, flow_calib, flow_test, r_calib, r_test,
                       p=1/288, n_boot=1000):
    """P1-8 fixed version: resample window blocks first, then compute the severity pooled coverage diff by flow tercile within each replicate."""
    vt = tt > 0
    _, q_u = cf.unconditional(pv, tv, pt, tt, labels, ALPHA, valid_val=tv > 0, valid_test=vt)
    # hard severity q
    _, meta = cf.mondrian_hard(pv, tv, pt, tt, labels, r_calib, r_test, K=5,
                               alpha=ALPHA, valid_val=tv > 0, valid_test=vt)
    edges = np.array(meta['edges']); qb = np.array(meta['q_bins'])
    ct = np.broadcast_to(np.asarray(r_test)[:, None, :], pt.shape)
    b = np.clip(np.digitize(ct, edges[1:-1]), 0, 4)
    q_s = qb[b]
    # flow tercile boundaries (calib-only, train-only spirit)
    f_flat = flow_calib[:pv.shape[0]].ravel()
    fq = np.quantile(f_flat, [1/3, 2/3])
    # per-window stats (uncond aligned with severity)
    n_win = pt.shape[0]
    cov_u = (tt >= pt - q_u) & (tt <= pt + q_u) & vt
    cov_s = (tt >= pt - q_s) & (tt <= pt + q_s) & vt
    flow_te = flow_test[:n_win]
    out = {}
    rng = np.random.default_rng(SEED)
    for si, sl in [(0, 'low'), (1, 'mid'), (2, 'high')]:
        if si == 0:
            sel = flow_te <= fq[0]
        elif si == 1:
            sel = (flow_te > fq[0]) & (flow_te <= fq[1])
        else:
            sel = flow_te > fq[1]
        sel3 = np.broadcast_to(sel[:, None, :], cov_u.shape)
        nu = (cov_u & sel3).sum(axis=(1, 2)).astype(np.float64)
        du = sel3.sum(axis=(1, 2)).astype(np.float64)
        ns = (cov_s & sel3).sum(axis=(1, 2)).astype(np.float64)
        diffs = np.empty(n_boot)
        for b_ in range(n_boot):
            idx = np.empty(n_win, dtype=np.int64)
            idx[0] = rng.integers(n_win)
            for i in range(1, n_win):
                idx[i] = rng.integers(n_win) if rng.random() < p else (idx[i-1]+1) % n_win
            duu = nu[idx].sum()/max(du[idx].sum(), 1)
            dss = ns[idx].sum()/max(du[idx].sum(), 1)
            diffs[b_] = dss - duu
        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        out[sl] = {'diff_mean': float(diffs.mean()), 'ci95': [float(lo), float(hi)]}
        print(f'  [P1-8] flow={sl:5s}: severity-uncond TypeC-cov diff={diffs.mean():+.4f} '
              f'CI=[{lo:+.4f},{hi:+.4f}] {"✓" if lo > 0 else "✗"}', flush=True)
    return out


def main():
    os.makedirs(EXP, exist_ok=True)
    print('[P1-7/P1-8] node-window rho + flow-stratified bootstrap (R3 scratch s42)', flush=True)
    dataset = 'PEMS04'; seed = 42
    full, shape = r3_eval.load_full(dataset)
    T = shape[0]; N = 307
    train_end, select_end, calib_end = int(T*0.6), int(T*0.7), int(T*0.8)
    tau = r3_eval.tx.fit_per_node_q95(full[:train_end, :, 0])
    stats = sev.fit_severity_stats(full[:train_end, :, 0])
    s_series = sev.severity_series(full[:, :, 0], stats)
    g_calib = np.arange(select_end, calib_end - 23)
    g_test = np.arange(calib_end, T - 23)
    r_c = sev.window_in_severity(s_series, g_calib, 12)
    r_t = sev.window_in_severity(s_series, g_test, 12)
    f_c = full[g_calib[:, None] + np.arange(12)][:, :, :, 0].mean(axis=1)
    f_t = full[g_test[:, None] + np.arange(12)][:, :, :, 0].mean(axis=1)
    labels = r3_eval.tx.classify_from_flow(full[:, :, 0], g_test, tau, 12, 12)
    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')
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
    rc, rt = r_c[:pv.shape[0]], r_t[:n]
    fc, ft = f_c[:pv.shape[0]], f_t[:n]

    rho = p1_rho_main(full, N, train_end, select_end, calib_end, stats, tau, lab,
                      pv, tv, pt, tt, rc, rt)
    strat = p1_flow_stratified(pv, tv, pt, tt, lab, fc, ft, rc, rt)
    out = {'date': '2026-08-18', 'dataset': dataset, 'seed': seed,
           'p1_7_node_rho': rho, 'p1_8_flow_stratified': strat}
    fn = os.path.join(EXP, 'metrics_p1_rho_flowstrat.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nsaved -> {fn}', flush=True)


if __name__ == '__main__':
    main()
