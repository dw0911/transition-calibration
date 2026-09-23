# -*- coding: utf-8 -*-
"""R2-b（决定性检验）：同容量 base vs base+severity —— severity 是否还有独立增量？

协议 v3 §5.1 的构造，全部落在缓存上，不新增点预测训练：
  a_{n,h}      Selection 上拟合的节点-预测步尺度（正值下限 + 稀疏节点回退到节点中位尺度）
  R_{i,n}      = max_h |Y-Ŷ| / a_{n,h}                    路径风险，单元 = (node, window)
  Z0           预测幅度/预测形状/历史流量/节点尺度/时间编码（全部部署前可得）
  Z1           = (Z0, S0)   S0 = r_in_max（冻结的双侧 robust max severity）
  g_j          HistGradientBoostingRegressor(loss=quantile, quantile=0.9)，两套同参数同样本
  q_j          Calibration 上 conformal_quantile(R / max(g_j, eps))，秩修正
  I^{(j)}      ŷ ± q_j · a_{n,h} · max(g_j, eps)

严格禁止：用 test 拟合尺度/挑分箱/选模型；用未来真值、TypeB/TypeC 标签、未成熟残差。
主判据（v3 §6.1，事前登记）：候选与参照经验路径覆盖均 ≥0.89；候选相对参照的覆盖差
  单侧 95% 下界 ≥ −0.01；该条件下平均宽度至少降低 5% 且配对 CI 支持改善。
主对比 = g1 vs g0（severity 的独立增量）；辅对比 = 两者 vs nh_norm_max / path_global。
用法: python r2b_incremental_severity.py [dataset seed]
"""
import os
import sys
import json
import time

_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
UC = os.path.join(PROJ, 'uncalib_v1')
ROOT = os.path.join(PROJ, 'revision_v1')
sys.path.insert(0, os.path.join(UC, 'SRC'))
import numpy as np                                    # noqa: E402
import conformal as cf                                # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402

ALPHA, K, H = 0.10, 5, 12
NOM = 1 - ALPHA
SPD = 288
EPS_REL = 1e-3
G_FLOOR_REL = 0.01           # g 的正值下限 = 1% × median(g_fit)
GROUPS = {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}
SEEDS = [('PEMS04', 42), ('PEMS04', 123), ('PEMS04', 456), ('PEMS03', 42)]
T_OF = {'PEMS04': 16992, 'PEMS03': 26208}
N_BOOT, P_RESTART = 2000, 1.0 / 288.0
LEARNER = dict(loss='quantile', quantile=0.90, learning_rate=0.05, max_iter=200,
               max_leaf_nodes=15, min_samples_leaf=100, early_stopping=False)


def load(ds, seed, tag):
    z = np.load(os.path.join(ROOT, 'cache', f'{ds}_s{seed}_{tag}.npz'))
    return {k: z[k] for k in z.files}


def g_ranges(ds):
    T = T_OF[ds]
    tr, se, ca = int(T * .6), int(T * .7), int(T * .8)
    return {'sel': np.arange(tr, se - 23), 'cal': np.arange(se, ca - 23),
            'test': np.arange(ca, T - 23)}


def fit_scale(dsel):
    """a_{n,h}：Selection 上的平均绝对残差（train-period 统计，含正值下限与稀疏回退）。"""
    R = np.abs(dsel['pred'] - dsel['true']).astype(np.float64)
    v = dsel['true'] > 0
    a = np.nanmean(np.where(v, R, np.nan), axis=0)                 # (H,N)
    med = np.nanmedian(np.where(np.isfinite(a), a, np.nan), axis=0)
    med = np.where(np.isfinite(med) & (med > 0), med, 1.0)
    bad = ~np.isfinite(a) | (a <= EPS_REL * med[None, :])
    a = np.where(bad, med[None, :], a)
    return a, int(bad.sum())


def features(d, g, a, with_sev):
    """(n,N) 场 → (n*N, F) 设计矩阵。全部字段仅依赖输入窗口与模型输出。"""
    p = d['pred'].astype(np.float64)                               # (n,H,N)
    n, _, N = p.shape
    mag = np.abs(p).mean(axis=1)
    std = p.std(axis=1)
    trend = (p[:, -3:, :].mean(axis=1) - p[:, :3, :].mean(axis=1)) / np.maximum(mag, 1.0)
    apos = p.argmax(axis=1).astype(np.float64) / (H - 1)
    node_scale = np.log(np.maximum(a.mean(axis=0), 1e-9))[None, :].repeat(n, 0)
    tod = ((g + H) % SPD) / SPD
    dow = ((g // SPD) % 7) / 7.0
    cols = [mag, std, trend, apos, d['flow_in'].astype(np.float64), node_scale,
            np.sin(2 * np.pi * tod)[:, None].repeat(N, 1), np.cos(2 * np.pi * tod)[:, None].repeat(N, 1),
            dow[:, None].repeat(N, 1)]
    if with_sev:
        cols.append(d['r_in_max'].astype(np.float64))
    X = np.stack([c.reshape(-1) for c in cols], axis=1).astype(np.float32)
    return X


def path_risk(d, a):
    """R_{i,n} = max_h |Y-Ŷ| / a_{n,h}；归一化在 max 之内完成。"""
    R = np.abs(d['pred'] - d['true']).astype(np.float64)
    v = (d['true'] > 0).all(axis=1)                                # V_{i,n}=1
    return (R / a[None, :, :]).max(axis=1)[v], v


def suff_from_q(q_half, pt, tt, lab, vt):
    """q_half: (n,H,N) 半宽。逐点与路径均限制在 V=1 群体。"""
    pt64, tt64 = pt.astype(np.float64), tt.astype(np.float64)
    lo, hi = pt64 - q_half, pt64 + q_half
    cov = (tt64 >= lo) & (tt64 <= hi)
    allv = vt.all(axis=1)
    width = 2.0 * q_half
    pen = (2.0 / ALPHA) * np.where(tt64 < lo, lo - tt64, 0.0) + \
        (2.0 / ALPHA) * np.where(tt64 > hi, tt64 - hi, 0.0)
    s = {'path_hit': (cov.all(axis=1) & allv).sum(axis=1).astype(np.float64),
         'path_elig': allv.sum(axis=1).astype(np.float64),
         'pw_cov': (cov & allv[:, None, :]).sum(axis=(1, 2)).astype(np.float64),
         'pw_elig': (np.broadcast_to(allv[:, None, :], cov.shape) & vt).sum(axis=(1, 2)).astype(np.float64),
         'w_sum': (width * vt).sum(axis=(1, 2)), 'w_n': vt.sum(axis=(1, 2)).astype(np.float64),
         'is_sum': ((width + pen) * vt).sum(axis=(1, 2))}
    for tid, gn in GROUPS.items():
        sel = (lab == tid) & allv
        s[f'{gn}_hit'] = (cov.all(axis=1) & sel).sum(axis=1).astype(np.float64)
        s[f'{gn}_elig'] = sel.sum(axis=1).astype(np.float64)
    return s


def rat(s, a, b):
    d = s[b].sum()
    return float(s[a].sum() / d) if d else float('nan')


def stationary_index(n_win, rng):
    idx = np.empty(n_win, dtype=np.int64); filled = 0
    while filled < n_win:
        st = int(rng.integers(n_win)); L = int(rng.geometric(P_RESTART))
        blk = (st + np.arange(L)) % n_win; take = min(L, n_win - filled)
        idx[filled:filled + take] = blk[:take]; filled += take
    return idx


def nh_norm_and_global(dsel, dcal, dtest, a):
    """两个强简单参照：nh_norm_max 与 path_global（与 r2a 同定义）。"""
    Rc = np.abs(dcal['pred'] - dcal['true']).astype(np.float64)
    allv_c = (dcal['true'] > 0).all(axis=1)
    q_glob = cf.conformal_quantile(Rc.max(axis=1)[allv_c], ALPHA)
    Sc = (Rc / a[None, :, :]).max(axis=1)[allv_c]
    q_n = cf.conformal_quantile(Sc, ALPHA)
    return q_glob, q_n


def run(ds, seed, out):
    t0 = time.time()
    dsel, dcal, dtest = load(ds, seed, 'sel'), load(ds, seed, 'cal'), load(ds, seed, 'test')
    g = g_ranges(ds)
    for tag, d in (('sel', dsel), ('cal', dcal), ('test', dtest)):
        assert d['pred'].shape[0] == len(g[tag]), f'{tag} 窗口数与边界不符'
    a, n_fb = fit_scale(dsel)
    pt, tt, lab = dtest['pred'], dtest['true'], dtest['labels']
    vt = tt > 0
    n, _, N = pt.shape

    Xs = {0: features(dsel, g['sel'], a, False), 1: features(dsel, g['sel'], a, True)}
    Xc = {0: features(dcal, g['cal'], a, False), 1: features(dcal, g['cal'], a, True)}
    Xt = {0: features(dtest, g['test'], a, False), 1: features(dtest, g['test'], a, True)}
    ys, vs = path_risk(dsel, a)
    yc, vc = path_risk(dcal, a)
    vs_f, vc_f = vs.ravel(), vc.ravel()          # X 的行序 = (n,N) 展平，与 v 对齐

    S, meta = {}, {'n_sel_rows': int(vs.sum()), 'n_cal_rows': int(vc.sum()),
                   'a_fallback_cells': n_fb, 'learner': {k: str(v) for k, v in LEARNER.items()},
                   'sklearn': __import__('sklearn').__version__}
    for j in (0, 1):
        mdl = HistGradientBoostingRegressor(random_state=seed, **LEARNER).fit(Xs[j][vs_f], ys)
        gc_ = np.maximum(mdl.predict(Xc[j]), 0.0).reshape(dcal['pred'].shape[0], N)
        gt_ = np.maximum(mdl.predict(Xt[j]), 0.0).reshape(n, N)
        floor = G_FLOOR_REL * float(np.median(gc_[vc]))
        gc_, gt_ = np.maximum(gc_, floor), np.maximum(gt_, floor)
        q_j = cf.conformal_quantile((yc / gc_[vc]), ALPHA)
        half = q_j * a[None, :, :] * gt_[:, None, :]
        S[f'g{j}_base' if j == 0 else 'g1_base_plus_severity'] = suff_from_q(half, pt, tt, lab, vt)
        meta[f'g{j}'] = {'q': q_j, 'g_floor': floor, 'median_g_cal': float(np.median(gc_[vc]))}
        del half, gc_, gt_
    q_glob, q_n = nh_norm_and_global(dsel, dcal, dtest, a)
    S['path_global'] = suff_from_q(np.full(pt.shape, q_glob), pt, tt, lab, vt)
    S['nh_norm_max'] = suff_from_q(q_n * a[None, :, :], pt, tt, lab, vt)
    meta['path_global'] = {'q': q_glob}; meta['nh_norm_max'] = {'q': q_n}

    metrics = {m: {'path_cov': rat(s, 'path_hit', 'path_elig'),
                   'pointwise_same_pop': rat(s, 'pw_cov', 'pw_elig'),
                   'mean_width': rat(s, 'w_sum', 'w_n'), 'IS': rat(s, 'is_sum', 'w_n'),
                   'n_paths': float(s['path_elig'].sum()),
                   'group_path_cov': {gn: rat(s, f'{gn}_hit', f'{gn}_elig') for gn in GROUPS.values()}}
               for m, s in S.items()}
    rng = np.random.default_rng(7)
    pairs = [('g1_base_plus_severity', 'g0_base'), ('g1_base_plus_severity', 'nh_norm_max'),
             ('g0_base', 'nh_norm_max'), ('nh_norm_max', 'path_global'),
             ('g1_base_plus_severity', 'path_global')]
    contr = {}
    for m, r in pairs:
        dc, wr = [], []
        for _ in range(N_BOOT):
            bi = stationary_index(n, rng)
            Sm = {k: v[bi] for k, v in S[m].items()}; Sr = {k: v[bi] for k, v in S[r].items()}
            dc.append(rat(Sm, 'path_hit', 'path_elig') - rat(Sr, 'path_hit', 'path_elig'))
            wr.append(rat(Sm, 'w_sum', 'w_n') / rat(Sr, 'w_sum', 'w_n'))
        dc, wr = np.array(dc), np.array(wr)
        contr[f'{m}__vs__{r}'] = {
            'd_path_cov': float(dc.mean()), 'd_CI95': [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
            'd_one_sided95_lower': float(np.percentile(dc, 5.0)),
            'width_ratio': float(wr.mean()), 'w_CI95': [float(np.percentile(wr, 2.5)), float(np.percentile(wr, 97.5))],
            'gate_pass': bool(metrics[m]['path_cov'] >= 0.89 and metrics[r]['path_cov'] >= 0.89
                              and np.percentile(dc, 5.0) >= -0.01 and wr.mean() <= 0.95)}
    out[f'{ds}_s{seed}'] = {'metrics': metrics, 'meta': meta, 'contrasts': contr,
                            'runtime_sec': round(time.time() - t0, 1)}
    print(f'\n===== {ds} s{seed} ({time.time()-t0:.0f}s) rows sel/cal='
          f'{meta["n_sel_rows"]}/{meta["n_cal_rows"]} a_fallback={n_fb}', flush=True)
    print(f'{"method":24s} {"pathCov":>8s} {"pw same":>8s} {"width":>8s} {"IS":>8s} '
          f'{"pathB":>7s} {"pathC":>7s} {"pathN":>7s}', flush=True)
    for m in ('path_global', 'nh_norm_max', 'g0_base', 'g1_base_plus_severity'):
        c = metrics[m]
        print(f'{m:24s} {c["path_cov"]:8.4f} {c["pointwise_same_pop"]:8.4f} {c["mean_width"]:8.1f} '
              f'{c["IS"]:8.1f} {c["group_path_cov"]["TypeB"]:7.3f} {c["group_path_cov"]["TypeC"]:7.3f} '
              f'{c["group_path_cov"]["Normal"]:7.3f}', flush=True)
    for k, v in contr.items():
        print(f'   {k:46s} Δcov={v["d_path_cov"]:+.4f} '
              f'1s95low={v["d_one_sided95_lower"]:+.4f} wratio={v["width_ratio"]:.4f} '
              f'CI=[{v["w_CI95"][0]:.4f},{v["w_CI95"][1]:.4f}] gate={"PASS" if v["gate_pass"] else "FAIL"}',
              flush=True)


def main():
    args = sys.argv[1:]
    seeds = [(args[0], int(args[1]))] if args else SEEDS
    out = {'protocol': 'v3 §5.1/§6.1; Z0=scale+shape+flow+node-scale+time; Z1=Z0+S0(r_in_max); '
           'matched learner/budget/seed; a_{n,h} and g fitted on Selection; q on Calibration; '
           'no test-side tuning', 'gate': 'path_cov>=0.89 both, one-sided95 lower of Δcov >= -0.01, '
           'width ratio <= 0.95', 'configs': {}}
    for ds, sd in seeds:
        run(ds, sd, out['configs'])
    fn = os.path.join(ROOT, 'results', 'r2b_incremental_severity.json')
    json.dump(out, open(fn, 'w', encoding='utf-8'), indent=1)
    print('\nsaved ->', fn)


if __name__ == '__main__':
    main()
