# -*- coding: utf-8 -*-
"""E5：自适应度—漂移惩罚的关联检验 → results/phase1_R1/drift_penalty.json

主张（PLAN_FROZEN §3.4 / claims_to_evidence A9，当前 PARTIAL）：
  所有方法都在同一 Calibration 段按名义 0.90 校准，但 test 达成覆盖随自适应度单调下降；
  条件化省下的宽度，部分由对 calib→test 漂移的脆弱性支付。

设计要点（避免只有 4 个方法点的低功效秩相关）：
  1) **连续自适应度旋钮 λ**：把任一方法的 per-(node,window) 尺度因子 f 向常数 1 收缩，
        f_λ = (1-λ) + λ·f/mean_calib(f)
     半宽 = q_λ · f_λ · u_{n,h}，其中 u 为该方法的元素形状函数（global/Mondrian: u=1；
     nh_norm/g0/g1: u=a_{n,h}）。q_λ 在 Calibration 上按名义 α 重新拟合，
     因此 λ=0 与 λ=1 都精确校准于 calib，唯一变化的是自适应度。
  2) **K 扫描**：路径级 Mondrian 的 K ∈ {1,3,5,7,10,20}，K=1 即非自适应。
  3) **漂移统计量**分两类并显式标注：
        label_free（部署时可得）：流量尺度比、severity 均值比、预测幅度比
        diagnostic（用到 test 标签，仅诊断）：残差尺度比、路径风险比
     二者不得混用；不得把 diagnostic 统计量当作可部署的漂移检测器。

统计功效声明（写死，不事后修饰）：
  - 剂量-反应（λ、K）在**每个 config 内部**有 5-6 个点，是本检验的主要证据；
  - 跨 config 的"漂移 × 自适应度"交互只有 4 个 config（2 数据集），**只能作提示性描述**，
    不得写成已确立的调节效应。
用法: python scripts/e5_drift_penalty.py
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
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np                                    # noqa: E402
import conformal as cf                                # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402
import r2b_incremental_severity as r2b                # noqa: E402

ALPHA, H = 0.10, 12
NOM = 1 - ALPHA
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
KGRID = [1, 3, 5, 7, 10, 20]
SEEDS = [('PEMS04', 42), ('PEMS04', 123), ('PEMS04', 456), ('PEMS03', 42)]


def load(ds, seed, tag):
    z = np.load(os.path.join(ROOT, 'cache', f'{ds}_s{seed}_{tag}.npz'))
    return {k: z[k] for k in z.files}


def drift_stats(dcal, dtest):
    """calib→test 漂移统计量。label_free 项部署可得；diagnostic 项用到 test 标签。"""
    vc = dcal['true'] > 0
    vt = dtest['true'] > 0
    Rc = np.abs(dcal['pred'] - dcal['true']).astype(np.float64)
    Rt = np.abs(dtest['pred'] - dtest['true']).astype(np.float64)
    allvc = vc.all(axis=1); allvt = vt.all(axis=1)
    out = {
        'label_free': {
            'flow_mean_ratio': float(dtest['flow_in'][vt.any(axis=1)].mean() /
                                     dcal['flow_in'][vc.any(axis=1)].mean()),
            'severity_mean_ratio': float(dtest['r_in_max'].mean() / dcal['r_in_max'].mean()),
            'pred_mag_ratio': float(dtest['pred_mag'].mean() / dcal['pred_mag'].mean()),
            'severity_q90_ratio': float(np.quantile(dtest['r_in_max'], 0.90) /
                                        max(np.quantile(dcal['r_in_max'], 0.90), 1e-12)),
        },
        'diagnostic_uses_test_labels': {
            'element_resid_scale_ratio': float(Rt[vt].mean() / Rc[vc].mean()),
            'path_risk_ratio': float(Rt.max(axis=1)[allvt].mean() / Rc.max(axis=1)[allvc].mean()),
            'raw_mae_calib': float(Rc[vc].mean()), 'raw_mae_test': float(Rt[vt].mean()),
        }}
    return out


def eval_halfwidth(half, pt, tt, lab=None):
    """给定 (n,H,N) 半宽，返回 test 上的路径同时覆盖、逐点覆盖（同 V=1 群体）、平均宽度、CV。"""
    pt64, tt64 = pt.astype(np.float64), tt.astype(np.float64)
    half = np.broadcast_to(np.asarray(half, dtype=np.float64), pt.shape)
    lo, hi = pt64 - half, pt64 + half
    cov = (tt64 >= lo) & (tt64 <= hi)
    vt = tt64 > 0
    allv = vt.all(axis=1)
    width = 2.0 * half
    return {'path_cov': float((cov.all(axis=1) & allv).sum() / max(allv.sum(), 1)),
            'pointwise_same_pop': float((cov & allv[:, None, :]).sum() /
                                        max(np.broadcast_to(allv[:, None, :], cov.shape)[vt].sum(), 1)),
            'mean_width': float(width[vt].mean())}


def path_scores(d, u):
    """路径风险 S_{i,n} = max_h |R_{i,n,h}| / u_{n,h}；u 为 (H,N) 或标量 1。"""
    R = np.abs(d['pred'] - d['true']).astype(np.float64)
    if np.ndim(u) == 2:
        R = R / u[None, :, :]
    v = (d['true'] > 0).all(axis=1)
    return R.max(axis=1), v


def cv_of_half(half, valid):
    """自适应度 = 校准段上半宽场的变异系数（对所有方法族同一定义）。"""
    half = np.broadcast_to(np.asarray(half, dtype=np.float64), valid.shape)
    h = half[valid]
    m = float(h.mean())
    return float(h.std() / m) if m > 0 else 0.0


def lam_row(f_cal, f_test, Sc, vc, St, vt, pt, tt, u, lam, dcal):
    """按 λ 收缩 f，在 calib 上重新拟合 q，评价 test。f 为 (n,N)。"""
    mref = float(f_cal[vc].mean())
    fcl = (1 - lam) + lam * (f_cal / mref)
    fts = (1 - lam) + lam * (f_test / mref)
    q = cf.conformal_quantile((Sc / fcl)[vc], ALPHA)
    ub = u[None, :, :] if np.ndim(u) == 2 else 1.0
    half_cal = q * fcl[:, None, :] * ub
    half_test = q * fts[:, None, :] * ub
    r = eval_halfwidth(half_test, pt, tt)
    r['lambda'] = lam
    r['q'] = float(q)
    r['cv_halfwidth_calib'] = cv_of_half(half_cal, (dcal['true'] > 0))
    r['shortfall_vs_nominal'] = NOM - r['path_cov']
    del half_cal, half_test
    return r


def mondrian_f(cv_cal, cv_test, Sc, vc, K):
    """路径级 Mondrian 的尺度因子 f = q_bin / mean(q_bin)（K=1 时 f≡1）。"""
    if K <= 1:
        return np.ones_like(cv_cal), np.ones_like(cv_test)
    edges = np.quantile(cv_cal[vc], np.linspace(0, 1, K + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    b = np.digitize(cv_cal[vc], edges[1:-1])
    q_all = cf.conformal_quantile(Sc[vc], ALPHA)
    qb = np.zeros(K)
    for k in range(K):
        m = b == k
        qb[k] = cf.conformal_quantile(Sc[vc][m], ALPHA) if m.sum() > 10 else q_all
    return qb[np.digitize(cv_cal, edges[1:-1])], qb[np.clip(np.digitize(cv_test, edges[1:-1]), 0, K - 1)]


def run(ds, seed, out):
    t0 = time.time()
    dsel, dcal, dtest = load(ds, seed, 'sel'), load(ds, seed, 'cal'), load(ds, seed, 'test')
    g = r2b.g_ranges(ds)
    a, _ = r2b.fit_scale(dsel)
    pt, tt = dtest['pred'], dtest['true']
    n, _, N = pt.shape
    ones = np.ones((n, N)); ones_c = np.ones((dcal['pred'].shape[0], N))

    # 路径风险（u=1 与 u=a 两套）
    Sc1_c, vc = path_scores(dcal, 1.0)
    Sc1_t, vt = path_scores(dtest, 1.0)
    Sca_c, _ = path_scores(dcal, a)
    Sca_t, _ = path_scores(dtest, a)

    entry = {'drift': drift_stats(dcal, dtest), 'methods': {}, 'lambda_sweeps': {}, 'K_sweeps': {}}

    # --- 非自适应参照 ---
    q_g = cf.conformal_quantile(Sc1_c[vc], ALPHA)
    half_g = np.full(pt.shape, q_g, dtype=np.float64)
    entry['methods']['path_global'] = eval_halfwidth(half_g, pt, tt)
    entry['methods']['path_global'].update(cv_halfwidth_calib=0.0, q=q_g)  # 常数半宽 → CV=0
    entry['methods']['path_global']['shortfall_vs_nominal'] = \
        NOM - entry['methods']['path_global']['path_cov']
    q_a = cf.conformal_quantile(Sca_c[vc], ALPHA)
    entry['methods']['nh_norm_max'] = eval_halfwidth(q_a * a[None, :, :], pt, tt)
    entry['methods']['nh_norm_max'].update(
        cv_halfwidth_calib=float(a.std() / a.mean()), q=q_a)   # half = q_a * a_{n,h}，与窗口无关
    entry['methods']['nh_norm_max']['shortfall_vs_nominal'] = \
        NOM - entry['methods']['nh_norm_max']['path_cov']
    del half_g

    # --- Mondrian severity（u=1）λ 扫描 ---
    f_c, f_t = mondrian_f(dcal['r_in_max'].astype(np.float64), dtest['r_in_max'].astype(np.float64),
                          Sc1_c, vc, 5)
    entry['lambda_sweeps']['path_sev_K5'] = [
        lam_row(f_c, f_t, Sc1_c, vc, Sc1_t, vt, pt, tt, 1.0, L, dcal) for L in LAMBDAS]

    # --- 学习式风险模型 g0 / g1（u=a）λ 扫描 ---
    Xs = {0: r2b.features(dsel, g['sel'], a, False), 1: r2b.features(dsel, g['sel'], a, True)}
    Xc = {0: r2b.features(dcal, g['cal'], a, False), 1: r2b.features(dcal, g['cal'], a, True)}
    Xt = {0: r2b.features(dtest, g['test'], a, False), 1: r2b.features(dtest, g['test'], a, True)}
    ys, vs = r2b.path_risk(dsel, a)
    vs_f = vs.ravel()
    for j, nm in ((0, 'g0_base'), (1, 'g1_base_plus_severity')):
        mdl = HistGradientBoostingRegressor(random_state=seed, **r2b.LEARNER).fit(Xs[j][vs_f], ys)
        gc_ = np.maximum(mdl.predict(Xc[j]), 0.0).reshape(dcal['pred'].shape[0], N)
        gt_ = np.maximum(mdl.predict(Xt[j]), 0.0).reshape(n, N)
        floor = r2b.G_FLOOR_REL * float(np.median(gc_[vc]))
        gc_, gt_ = np.maximum(gc_, floor), np.maximum(gt_, floor)
        fjc, fjt = gc_ / float(np.median(gc_[vc])), gt_ / float(np.median(gc_[vc]))
        entry['lambda_sweeps'][nm] = [
            lam_row(fjc, fjt, Sca_c, vc, Sca_t, vt, pt, tt, a, L, dcal) for L in LAMBDAS]
        entry['methods'][nm] = dict(entry['lambda_sweeps'][nm][-1])
        del mdl, gc_, gt_

    # --- K 扫描（路径级 severity Mondrian，u=1）---
    for K in KGRID:
        fk_c, fk_t = mondrian_f(dcal['r_in_max'].astype(np.float64),
                                dtest['r_in_max'].astype(np.float64), Sc1_c, vc, K)
        q = cf.conformal_quantile((Sc1_c / fk_c)[vc], ALPHA)
        r = eval_halfwidth(q * fk_t[:, None, :], pt, tt)
        r['K'] = K
        r['cv_halfwidth_calib'] = cv_of_half(q * fk_c[:, None, :], dcal['true'] > 0)
        r['shortfall_vs_nominal'] = NOM - r['path_cov']
        entry['K_sweeps'][f'K={K}'] = r

    # --- 汇总：自适应度 vs 达成覆盖 ---
    pts = []
    for nm, rows in entry['lambda_sweeps'].items():
        for r in rows:
            pts.append({'family': nm, 'dial': f'lambda={r["lambda"]}',
                        'cv': r['cv_halfwidth_calib'], 'shortfall': r['shortfall_vs_nominal'],
                        'path_cov': r['path_cov'], 'width': r['mean_width']})
    for k, r in entry['K_sweeps'].items():
        pts.append({'family': 'path_sev_Ksweep', 'dial': k, 'cv': r['cv_halfwidth_calib'],
                    'shortfall': r['shortfall_vs_nominal'], 'path_cov': r['path_cov'],
                    'width': r['mean_width']})
    entry['adaptivity_points'] = pts

    def assoc(cv_, sf_):
        cv_, sf_ = np.asarray(cv_, float), np.asarray(sf_, float)
        if len(cv_) < 3 or cv_.std() == 0:
            return {'n': int(len(cv_)), 'pearson': None, 'spearman': None, 'slope': None}
        rk = np.argsort(np.argsort(cv_)).astype(float)
        rs = np.argsort(np.argsort(sf_)).astype(float)
        return {'n': int(len(cv_)), 'pearson': float(np.corrcoef(cv_, sf_)[0, 1]),
                'spearman': float(np.corrcoef(rk, rs)[0, 1]),
                'slope': float(np.polyfit(cv_, sf_, 1)[0])}

    per_family = {}
    for nm, rows in entry['lambda_sweeps'].items():
        per_family[nm] = assoc([r['cv_halfwidth_calib'] for r in rows],
                               [r['shortfall_vs_nominal'] for r in rows])
    per_family['path_sev_Ksweep'] = assoc([r['cv_halfwidth_calib'] for r in entry['K_sweeps'].values()],
                                          [r['shortfall_vs_nominal'] for r in entry['K_sweeps'].values()])
    entry['within_config_association'] = {
        'per_family': per_family,
        'pooled_all_points': assoc([p['cv'] for p in pts], [p['shortfall'] for p in pts]),
        'note': 'cv = 校准段上半宽场的变异系数（自适应度，所有方法族同一定义）；'
                'shortfall = 0.90 - test 达成路径覆盖。pooled 各点高度重叠（同族不同 λ/K），'
                '有效样本量远小于点数，故以 per_family 的剂量-反应斜率为主证据'}
    out[f'{ds}_s{seed}'] = entry
    print(f'\n===== {ds} s{seed} ({time.time()-t0:.0f}s)', flush=True)
    lf = entry['drift']['label_free']; dg = entry['drift']['diagnostic_uses_test_labels']
    print(f"  漂移(label_free): flow={lf['flow_mean_ratio']:.4f} sev={lf['severity_mean_ratio']:.4f} "
          f"pmag={lf['pred_mag_ratio']:.4f} sev_q90={lf['severity_q90_ratio']:.4f}")
    print(f"  漂移(diagnostic): resid_scale={dg['element_resid_scale_ratio']:.4f} "
          f"path_risk={dg['path_risk_ratio']:.4f} (mae {dg['raw_mae_calib']:.2f}->{dg['raw_mae_test']:.2f})")
    for nm in ('path_global', 'nh_norm_max', 'g0_base', 'g1_base_plus_severity'):
        m = entry['methods'][nm]
        print(f"  {nm:22s} path={m['path_cov']:.4f} shortfall={m['shortfall_vs_nominal']:+.4f} "
              f"cv={m['cv_halfwidth_calib']:.4f} width={m['mean_width']:6.1f}")
    print('  λ 剂量-反应（g1）: ' + ' '.join(
        f"λ{r['lambda']:.2f}:cov{r['path_cov']:.4f}/w{r['mean_width']:.0f}/cv{r['cv_halfwidth_calib']:.3f}"
        for r in entry['lambda_sweeps']['g1_base_plus_severity']))
    print('  K 扫描: ' + ' '.join(f"K{r['K']}:{r['path_cov']:.4f}(cv{r['cv_halfwidth_calib']:.3f})"
                                  for r in entry['K_sweeps'].values()))
    for fam, st in entry['within_config_association']['per_family'].items():
        print(f"  {fam:24s} n={st['n']} rho={st['spearman']} slope={st['slope']}")


def main():
    out = {'protocol': 'lambda-shrinkage dial + K sweep; q 每次在 Calibration 上按名义 alpha 重拟合，'
           '故 lam=0 与 lam=1 均精确校准于 calib，唯一变化量是自适应度 cv(f)',
           'power_statement': '剂量-反应每 config 有 5-6 点（主证据）；跨 config 交互仅 4 个 config '
                              '（2 数据集），只作提示性描述，不得写成已确立的调节效应',
           'drift_stat_classes': {'label_free': '部署可得，不含 test 标签',
                                  'diagnostic_uses_test_labels': '仅诊断，不得当作可部署漂移检测器'},
           'configs': {}}
    for ds, sd in SEEDS:
        run(ds, sd, out['configs'])
    # 跨 config 汇总
    rows = []
    for c, e in out['configs'].items():
        d = e['drift']
        rows.append({'config': c,
                     'drift_path_risk_ratio': d['diagnostic_uses_test_labels']['path_risk_ratio'],
                     'drift_sev_ratio': d['label_free']['severity_mean_ratio'],
                     'shortfall_global': e['methods']['path_global']['shortfall_vs_nominal'],
                     'shortfall_g1': e['methods']['g1_base_plus_severity']['shortfall_vs_nominal'],
                     'spread': (e['methods']['path_global']['shortfall_vs_nominal'] -
                                e['methods']['g1_base_plus_severity']['shortfall_vs_nominal']),
                     'cv_global': e['methods']['path_global']['cv_halfwidth_calib'],
                     'cv_g1': e['methods']['g1_base_plus_severity']['cv_halfwidth_calib'],
                     'slopes': {f: (v['slope'], v['spearman'])
                                for f, v in e['within_config_association']['per_family'].items()}})
    out['cross_config'] = rows
    fn = os.path.join(ROOT, 'results', 'phase1_R1', 'drift_penalty.json')
    json.dump(out, open(fn, 'w', encoding='utf-8'), indent=1)
    print('\n===== 跨 config 汇总（提示性，n=4）=====')
    for r in rows:
        sl = r['slopes'].get('g1_base_plus_severity', (None, None))
        print(f"  {r['config']:14s} drift_path_risk={r['drift_path_risk_ratio']:.4f} "
              f"drift_sev={r['drift_sev_ratio']:.4f} | cv(global)={r['cv_global']:.3f} "
              f"cv(g1)={r['cv_g1']:.3f} | shortfall {r['shortfall_global']:+.4f}(global) -> "
              f"{r['shortfall_g1']:+.4f}(g1) spread={r['spread']:+.4f} | g1 λ-slope={sl[0]} rho={sl[1]}")
    if len(rows) > 2:
        for xname in ('drift_path_risk_ratio', 'drift_sev_ratio'):
            x = np.array([r[xname] for r in rows])
            y = np.array([r['spread'] for r in rows])
            print(f"  {xname} vs spread(global-g1): pearson={np.corrcoef(x, y)[0,1]:+.3f} "
                  f"(n={len(rows)}, 提示性，不得写成已确立的调节效应)")
        cx = np.array([r['cv_g1'] for r in rows]); cy = np.array([r['spread'] for r in rows])
        print(f"  cv(g1) vs spread: pearson={np.corrcoef(cx, cy)[0,1]:+.3f} (n={len(rows)})")
    print('\nsaved ->', fn)


if __name__ == '__main__':
    main()
