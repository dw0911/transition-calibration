# -*- coding: utf-8 -*-
"""R3 four-split canonical evaluation (post Round1-audit-fix main table).

Protocol (identical to train_r3.py):
  - four-split boundaries: train=int(T*0.6) / select=int(T*0.7) / calib=int(T*0.8) / test=T
  - window start g in half-open range [start, end-24) (window output end point <= end-1)
  - calib start = select_end (train select last g = select_end-24 -> implicitly purges 23 anchors, satisfies P1-6)
  - canonical taxonomy (12-step misalignment fix + zero mask) + unified severity (Q99 clip) + hard-bin main method
  - valid mask (true>0) threaded through coverage/WCE/GCD
  - alpha=0.10; outputs WCE/WCE_group/GCD/q_monotonic/per-group width + interp variant

Usage: python r3_eval.py
"""
import os
import sys
import json

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASICTS = os.environ.get('BASICTS_ROOT', '')
UC = REPRO_ROOT
EXP = os.path.join(UC, 'experiments', 'artifacts')

sys.path.insert(0, os.path.join(UC, 'SRC'))
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
os.chdir(BASICTS)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import conformal as cf                              # noqa: E402
import taxonomy as tx                               # noqa: E402
import severity as sev                              # noqa: E402

ALPHA = 0.10
IL, OL = 12, 12
DATASET_CFG = {'PEMS04': {'N': 307, 'seeds': [42, 123, 456]},
               'PEMS03': {'N': 358, 'seeds': [42]}}


def load_full(dataset):
    desc = json.load(open(os.path.join(BASICTS, 'datasets', dataset, 'desc.json')))
    shape = tuple(desc['shape'])
    return np.array(np.memmap(os.path.join(BASICTS, 'datasets', dataset, 'data.dat'),
                              dtype='float32', mode='r', shape=shape)), shape


def build_model(N):
    from baselines.STAEformer.arch.staeformer_arch import STAEformer
    return STAEformer(num_nodes=N, in_steps=12, out_steps=12, steps_per_day=288,
                      input_dim=3, output_dim=1, input_embedding_dim=24,
                      tod_embedding_dim=24, dow_embedding_dim=24,
                      spatial_embedding_dim=0, adaptive_embedding_dim=80,
                      feed_forward_dim=256, num_heads=4, num_layers=3,
                      dropout=0.1, use_mixed_proj=True)


@torch.no_grad()
def infer_range(model, full, g_start, g_end, mean_t, std_t, N, bs=64):
    """Window start g in [g_start, g_end). Returns pred/true (n,12,N) raw."""
    gs = np.arange(g_start, g_end)
    n = len(gs)
    preds = []
    for s in range(0, n, bs):
        e = min(s + bs, n)
        gi = gs[s:e]
        x = full[gi[:, None] + np.arange(IL)].copy()               # (B,12,N,3) raw
        xn = torch.tensor(x, dtype=torch.float32, device='cuda')
        xn[..., 0] = (xn[..., 0] - mean_t) / std_t                 # same GPU-side norm as train_r3
        fut = torch.zeros(e - s, OL, N, 3, device='cuda')
        with torch.amp.autocast('cuda', enabled=True):
            out = model(history_data=xn[:, :, :, [0, 1, 2]], future_data=fut,
                        batch_seen=s, epoch=1, train=False)
        if isinstance(out, dict):
            out = out['prediction']
        preds.append((out.squeeze(-1).float() * std_t + mean_t).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    yj = gs[:, None] + np.arange(IL, IL + OL)
    true = full[yj][..., 0]
    return pred, true


def main():
    os.makedirs(EXP, exist_ok=True)
    print('[R3-eval] start', flush=True)
    for dataset, cfg in DATASET_CFG.items():
        full, shape = load_full(dataset)
        T = shape[0]; N = cfg['N']
        train_end, select_end, calib_end = int(T * 0.6), int(T * 0.7), int(T * 0.8)
        print(f'\n=== {dataset} T={T} N={N} '
              f'boundaries=({train_end},{select_end},{calib_end},{T}) ===', flush=True)

        # canonical stats (train-only)
        tau = tx.fit_per_node_q95(full[:train_end, :, 0])
        stats = sev.fit_severity_stats(full[:train_end, :, 0])
        s_series = sev.severity_series(full[:, :, 0], stats)
        g_calib = np.arange(select_end, calib_end - IL - OL + 1)   # implicit purge 23
        g_test = np.arange(calib_end, T - IL - OL + 1)
        r_calib = sev.window_in_severity(s_series, g_calib, IL)
        r_test = sev.window_in_severity(s_series, g_test, IL)
        flow_calib = full[g_calib[:, None] + np.arange(IL)][:, :, :, 0].mean(axis=1)
        flow_test = full[g_test[:, None] + np.arange(IL)][:, :, :, 0].mean(axis=1)
        labels = tx.classify_from_flow(full[:, :, 0], g_test, tau, IL, OL)
        prev = {g: float((labels == k).mean()) for k, g in
                {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}.items()}
        print(f'  calib_windows={len(g_calib)} test_windows={len(g_test)} '
              f'prevalence={prev}', flush=True)

        mean = full[:train_end, :, 0].mean(axis=0)
        std = full[:train_end, :, 0].std(axis=0)
        std[std == 0] = 1.0
        mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
        std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

        for seed in cfg['seeds']:
            print(f'\n--- {dataset} seed{seed} ---', flush=True)
            ckpt = os.path.join(UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints',
                                f'{dataset}_r3_4split_seed{seed}', 'best.pt')
            model = build_model(N).cuda()
            sd = torch.load(ckpt, map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            model.load_state_dict(sd)
            model.eval()

            pv, tv = infer_range(model, full, select_end, calib_end - IL - OL + 1,
                                 mean_t, std_t, N)
            pt, tt = infer_range(model, full, calib_end, T - IL - OL + 1, mean_t, std_t, N)
            n = min(pt.shape[0], labels.shape[0])
            pt, tt, lab = pt[:n], tt[:n], labels[:n]
            rc, rt = r_calib[:pv.shape[0]], r_test[:n]
            fc, ft = flow_calib[:pv.shape[0]], flow_test[:n]
            pmc, pmt = np.abs(pv).mean(axis=1), np.abs(pt).mean(axis=1)
            vv, vt = tv > 0, tt > 0

            res_u, q = cf.unconditional(pv, tv, pt, tt, lab, ALPHA, valid_val=vv, valid_test=vt)
            out = {'unconditional': res_u, 'q': q,
                   'raw_mae': float(np.abs(pt - tt)[vt].mean())}
            for cname, (cvc, cvt) in {'severity': (rc, rt), 'flow_level': (fc, ft),
                                      'predicted_mag': (pmc, pmt)}.items():
                rh, mh = cf.mondrian_hard(pv, tv, pt, tt, lab, cvc, cvt, K=5,
                                          alpha=ALPHA, valid_val=vv, valid_test=vt)
                ri, _ = cf.mondrian_interpolated(pv, tv, pt, tt, lab, cvc, cvt, K=5,
                                                 alpha=ALPHA, valid_val=vv, valid_test=vt)
                out[cname] = {'hard': rh, 'interp': ri, 'meta': mh}
                dw = res_u['WCE'] - rh['WCE']; dg = res_u['GCD'] - rh['GCD']
                print(f"  [{cname} hard] TypeC={rh['TypeC']['cov']:.4f} "
                      f"WCE={res_u['WCE']:.4f} -> {rh['WCE']:.4f}({rh['WCE_group']}) "
                      f"deltaWCE={dw:+.4f} deltaGCD={dg:+.4f} width={res_u['mean_width']:.1f} -> {rh['mean_width']:.1f} "
                      f"mono={rh['q_monotonic']}", flush=True)
            print(f"  uncond: overall={res_u['overall_cov']:.4f} "
                  f"Normal={res_u['Normal']['cov']:.4f} TypeC={res_u['TypeC']['cov']:.4f} "
                  f"WCE={res_u['WCE']:.4f}({res_u['WCE_group']}) GCD={res_u['GCD']:.4f} "
                  f"raw_mae={out['raw_mae']:.3f} q={q:.2f}", flush=True)

            fn = os.path.join(EXP, f'metrics_r3_{dataset}_s{seed}.json')
            with open(fn, 'w', encoding='utf-8') as f:
                json.dump({'date': '2026-08-18', 'dataset': dataset, 'seed': seed,
                           'protocol': 'R3 four-split canonical (purge 23, hard-bin main, valid mask)',
                           'alpha': ALPHA, 'boundaries': (train_end, select_end, calib_end, T),
                           'prevalence': prev, 'results': out}, f, indent=2,
                          ensure_ascii=False, default=str)
            print(f'saved -> {fn}', flush=True)
    print('\n[R3-eval] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
