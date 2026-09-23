# -*- coding: utf-8 -*-
"""S0 预测缓存导出 —— revision_v1.

协议严格复刻 uncalib_v1/EXPERIMENTS/R2_recalc/r3_eval.py（R3 四段 canonical）：
  train=[0,int(T*.6)) / select=[train_end,select_end) / calib=[select_end,calib_end-23) / test=[calib_end,T-23)
  canonical taxonomy + 统一 severity(Q99 clip) + valid mask(true>0) + alpha=0.10

新增（r3_eval 未导出，修订版必需）：
  1) selection 段预测 —— 供 M06/M07 风险模型拟合（原计划 D2）
  2) 输入窗口 severity 剖面派生量 —— argmax 位置 / 距末步时滞 / 近端加权 severity
     用于 P3（TypeA 浪费与"位置感知 severity"），因为 r^in=max 对窗口内位置失忆

用法:
  python export_preds.py PEMS04 42
  python export_preds.py all
"""
import os
import sys
import json
import time

_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
BASICTS = os.path.join(PROJ, 'BasicTS058')
UC = os.path.join(PROJ, 'uncalib_v1')
ROOT = os.path.join(PROJ, 'revision_v1')
CACHE = os.path.join(ROOT, 'cache')

sys.path.insert(0, os.path.join(UC, 'SRC'))
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
os.chdir(BASICTS)

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import taxonomy as tx                                 # noqa: E402
import severity as sev                                # noqa: E402

ALPHA = 0.10
IL, OL = 12, 12
PURGE = IL + OL - 1          # 23：窗口输出末点 <= end-1
CFG = {'PEMS04': {'N': 307, 'seeds': [42, 123, 456]},
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
    """与 r3_eval.infer_range 逐行一致（保证缓存可复现已发表数字）。"""
    gs = np.arange(g_start, g_end)
    n = len(gs)
    preds = []
    for s in range(0, n, bs):
        e = min(s + bs, n)
        gi = gs[s:e]
        x = full[gi[:, None] + np.arange(IL)].copy()
        xn = torch.tensor(x, dtype=torch.float32, device='cuda')
        xn[..., 0] = (xn[..., 0] - mean_t) / std_t
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
    return pred.astype(np.float32), true.astype(np.float32)


def window_features(s_series, g_indices):
    """从 severity 剖面派生 (n_win, N) 条件变量。

    r_in_max : 发表版主方法条件变量（max over input window）
    argpos   : 达到 max 的差分位置 0..IL-2（0=最早）
    lag      : (IL-2) - argpos，越大=转变越早、越"已过去"
    r_last3  : 最后 3 个输入差分的 max（近端强度）
    r_recency: 指数衰减加权 max_j decay^(dist_j) * s_j  —— P3 的位置感知候选
    """
    g = np.asarray(g_indices)
    ar = np.arange(IL - 1)
    blk = s_series[g[:, None] + ar]                                 # (n, IL-1, N) float64
    r_max = blk.max(axis=1)
    amax = blk.argmax(axis=1)                                   # (n,N)
    lag = (IL - 2) - amax.astype(np.float32)
    r_last3 = blk[:, -3:, :].max(axis=1).astype(np.float32)
    decay = np.float64(0.6)
    w = decay ** ((IL - 2) - ar)[None, :, None]
    r_rec = (blk * w).max(axis=1).astype(np.float32)
    del blk
    return {'r_in_max': r_max, 'argpos': amax.astype(np.int8), 'lag': lag,
            'r_last3': r_last3, 'r_recency': r_rec}


def run(dataset, seed):
    t0 = time.time()
    full, shape = load_full(dataset)
    T = shape[0]; N = CFG[dataset]['N']
    train_end, select_end, calib_end = int(T * 0.6), int(T * 0.7), int(T * 0.8)
    g_sel = (train_end, select_end - PURGE)
    g_cal = (select_end, calib_end - PURGE)
    g_tst = (calib_end, T - PURGE)
    print(f'=== {dataset} seed{seed} T={T} N={N} '
          f'bnds=({train_end},{select_end},{calib_end},{T})', flush=True)

    flow = full[:, :, 0]
    tau = tx.fit_per_node_q95(full[:train_end, :, 0])
    stats = sev.fit_severity_stats(full[:train_end, :, 0])
    s_series = sev.severity_series(full[:, :, 0], stats)
    lab = tx.classify_from_flow(full[:, :, 0], np.arange(g_tst[0], g_tst[1]), tau, IL, OL)

    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')

    ckpt = os.path.join(UC, 'EXPERIMENTS', 'UC-G3', 'checkpoints',
                        f'{dataset}_r3_4split_seed{seed}', 'best.pt')
    model = build_model(N).cuda()
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd)
    model.eval()

    out = {'meta': json.dumps({'dataset': dataset, 'seed': seed, 'alpha': ALPHA,
                               'IL': IL, 'OL': OL, 'purge': PURGE,
                               'boundaries': [train_end, select_end, calib_end, T],
                               'g_ranges': [list(g_sel), list(g_cal), list(g_tst)],
                               'q99': stats.q99, 'note': 'sel/cal/test: pred,true,'
                               'r_in_max,flow_in,pred_mag; test also labels + severity pos feats'})}
    for tag, (a, b) in zip(('sel', 'cal', 'test'), (g_sel, g_cal, g_tst)):
        p, t = infer_range(model, full, a, b, mean_t, std_t, N)
        g = np.arange(a, b)
        fw = window_features(s_series, g)
        flow_in = full[g[:, None] + np.arange(IL)][:, :, :, 0].mean(axis=1).astype(np.float32)
        pmag = np.abs(p).mean(axis=1).astype(np.float32)
        arrs = {'pred': p, 'true': t, 'flow_in': flow_in, 'pred_mag': pmag}
        # dtype 与发表管线一致：r_in_max 走 float64（s_series 本身 float64），
        # pred/true/flow_in/pred_mag 走 float32 —— 保证复算逐位可对齐
        arrs['r_in_max'] = fw['r_in_max'].astype(np.float64)
        for k in ('argpos', 'lag', 'r_last3', 'r_recency'):
            arrs[k] = fw[k].astype(np.float32) if k != 'argpos' else fw[k]
        if tag == 'test':
            arrs['labels'] = lab
        np.savez(os.path.join(CACHE, f'{dataset}_s{seed}_{tag}.npz'), **arrs)
        print(f'  [{tag}] n={p.shape[0]} rawMAE='
              f'{np.abs(p - t)[t > 0].mean():.4f} ({time.time()-t0:.0f}s)', flush=True)
        del p, t, arrs
    torch.cuda.empty_cache()
    del model
    print(f'  done {dataset} s{seed} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    args = sys.argv[1:]
    if not args or args[0] == 'all':
        for ds, c in CFG.items():
            for s in c['seeds']:
                run(ds, s)
    else:
        run(args[0], int(args[1]))
