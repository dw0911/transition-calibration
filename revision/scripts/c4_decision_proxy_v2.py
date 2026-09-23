# -*- coding: utf-8 -*-
"""C4 决策代理（校审修正版 v2）。只读缓存，不重训。

作者校审（2026-09-21）指出的三处修正，全部落实：
  1) **概率口径**：同时报告三个量——认证比例 P(S)、以全部有效路径为分母的错误认证率
     P(A∩S)、认证后条件错误率 P(A|S)。**删除"FCR ≤ α"的理论预期**：由 A∩S ⊆ M 与
     P(M) ≤ α 只能得 P(A∩S) ≤ α 与 P(A|S) ≤ min(1, α/P(S))；实际交通数据用经验表述。
  2) **峰值得分**：新增真峰值残差基线 peak_true，得分 s = max_h Y − max_h Ŷ（对预测峰值
     直接校准）；现有 max_h(Y−Ŷ) 改名 stepwise_upper 保留（它承担逐步上界任务，对峰值更保守）。
  3) **峰值越带**：在统一 test 的 V=1 群体上直接统计"真实峰值 > 预测峰值上界"，
     不再用 calibration 段残差事件、不再混用不完整路径。

方法族：uncond / sev / flow / pmag / nh_elem / nh_path / g0 / g1 / stepwise_upper / peak_true。
k_n：selection 段 per-node 流量 Q95/Q99（代理量，非真实运营容量；见 claims C4 备注）。

输出: results/phase1_R1/c4_decision_proxy_v2.json
"""
import os
import sys
import json

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJ, 'uncalib_v1', 'SRC'))
os.chdir(ROOT)

import conformal as cf                                # noqa: E402
import r2b_incremental_severity as r2b                # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402

ALPHA, K, H = 0.10, 5, 12
OUT = os.path.join(ROOT, 'results', 'phase1_R1', 'c4_decision_proxy_v2.json')
CONFIGS = [('PEMS04', 42, 'stae'), ('PEMS04', 123, 'stae'), ('PEMS04', 456, 'stae'),
           ('PEMS03', 42, 'stae'), ('PEMS03', 123, 'stae'), ('PEMS03', 456, 'stae'),
           ('PEMS08', 42, 'stae'), ('PEMS08', 123, 'stae'), ('PEMS08', 456, 'stae'),
           ('PEMS08', 42, 'stid'), ('PEMS08', 123, 'stid'), ('PEMS08', 456, 'stid')]


def load(ds, seed, tag, bb):
    suf = '' if bb == 'stae' else '_stid'
    z = np.load(os.path.join(ROOT, 'cache', f'{ds}{suf}_s{seed}_{tag}.npz'))
    return {k: z[k] for k in z.files}


def g_ranges(ds):
    desc = json.load(open(os.path.join(PROJ, 'BasicTS058', 'datasets', ds, 'desc.json')))
    T = desc['shape'][0]
    tr, se, ca = int(T * .6), int(T * .7), int(T * .8)
    return {'sel': np.arange(tr, se - 23), 'cal': np.arange(se, ca - 23),
            'test': np.arange(ca, T - 23)}


def run(ds, seed, bb):
    name = f'{ds}_{bb}_s{seed}'
    dsel, dcal, dtest = load(ds, seed, 'sel', bb), load(ds, seed, 'cal', bb), \
        load(ds, seed, 'test', bb)
    dsel['dataset'] = ds
    pt, tt, lab = dtest['pred'], dtest['true'], dtest['labels']
    pc, tc = dcal['pred'], dcal['true']
    vc = tc > 0
    allv_c = vc.all(axis=1)
    Rc = np.abs(pc - tc).astype(np.float64)
    a, _ = r2b.fit_scale(dsel)
    allv = (tt > 0).all(axis=1)
    nV1 = int(allv.sum())

    q = {'uncond': np.full(pt.shape, cf.unconditional(pc, tc, pt, tt, lab, ALPHA,
                                                      valid_val=vc, valid_test=tt > 0)[1])}
    for nm, ck in (('sev', 'r_in_max'), ('flow', 'flow_in'), ('pmag', 'pred_mag')):
        _, m = cf.mondrian_hard(pc, tc, pt, tt, lab, dcal[ck].astype(np.float64),
                                dtest[ck].astype(np.float64), K=K, alpha=ALPHA,
                                valid_val=vc, valid_test=tt > 0)
        e, qb = np.asarray(m['edges']), np.asarray(m['q_bins'])
        b = np.clip(np.digitize(cf._broadcast_cond(dtest[ck].astype(np.float64), pt.shape),
                                e[1:-1]), 0, K - 1)
        q[nm] = qb[b]
    q['nh_elem'] = cf.conformal_quantile((Rc / a[None, :, :])[vc], ALPHA) * a[None, :, :]
    q['nh_path'] = cf.conformal_quantile((Rc / a[None, :, :]).max(axis=1)[allv_c], ALPHA) \
        * a[None, :, :]
    g = g_ranges(ds)
    n_c, n_t, N = dcal['pred'].shape[0], pt.shape[0], a.shape[1]
    Xs = {j: r2b.features(dsel, g['sel'], a, j == 1) for j in (0, 1)}
    Xc = {j: r2b.features(dcal, g['cal'], a, j == 1) for j in (0, 1)}
    Xt = {j: r2b.features(dtest, g['test'], a, j == 1) for j in (0, 1)}
    ys, vs = r2b.path_risk(dsel, a)
    yc, vcc = r2b.path_risk(dcal, a)
    for j, nm in ((0, 'g0'), (1, 'g1')):
        mdl = HistGradientBoostingRegressor(random_state=seed, **r2b.LEARNER).fit(
            Xs[j][vs.ravel()], ys)
        gc_ = np.maximum(mdl.predict(Xc[j]), 0.0).reshape(n_c, N)
        gt_ = np.maximum(mdl.predict(Xt[j]), 0.0).reshape(n_t, N)
        floor = r2b.G_FLOOR_REL * float(np.median(gc_[vcc]))
        gc_, gt_ = np.maximum(gc_, floor), np.maximum(gt_, floor)
        qj = cf.conformal_quantile((yc / gc_[vcc]), ALPHA)
        q[nm] = qj * a[None, :, :] * gt_[:, None, :]

    peak_pred = pt.max(axis=1)
    peak_true_test = tt.max(axis=1)
    q_step = cf.conformal_quantile((tc - pc).max(axis=1)[allv_c], ALPHA)
    q_peaktrue = cf.conformal_quantile(
        (tc.max(axis=1) - pc.max(axis=1))[allv_c], ALPHA)

    res = {'config': name, 'n_V1': nV1, 'thresholds': {}}
    for ql in (0.95, 0.99):
        pool = dsel['true'].reshape(-1, dsel['true'].shape[-1])
        mask = pool > 0
        kn = np.array([np.quantile(pool[mask[:, j], j], ql) for j in range(pool.shape[1])])
        safe = (peak_true_test <= kn[None, :]) & allv
        block = {'k_node_source': f'selection-segment per-node flow Q{ql:.2f} (proxy)',
                 'n_safe': int(safe.sum()), 'methods': {}}
        for m, qq in list(q.items()) + [('stepwise_upper', None), ('peak_true', None)]:
            if m == 'stepwise_upper':
                ub = peak_pred + q_step
            elif m == 'peak_true':
                ub = peak_pred + q_peaktrue
            else:
                ub = (pt + np.broadcast_to(np.asarray(qq, np.float64), pt.shape)).max(axis=1)
            certify = (ub <= kn[None, :]) & allv
            nS = int(certify.sum())
            err = int((certify & ~safe).sum())
            exceed = int(((peak_true_test > ub) & allv).sum())
            block['methods'][m] = {
                'P_S': nS / nV1,
                'P_A_and_S': err / nV1,
                'P_A_given_S': (err / nS) if nS else float('nan'),
                'bound_P_A_given_S': min(1.0, ALPHA / (nS / nV1)) if nS else float('nan'),
                'peak_exceedance_V1': exceed / nV1}
        res['thresholds'][f'Q{ql:.2f}'] = block
    return res


def main():
    rep = {'protocol': 'C4 v2 (author-audit corrected): reports P(S), P(A∩S), P(A|S) and the '
                       'bound min(1, α/P(S)); no claim that P(A|S) ≤ α. peak_true uses score '
                       'max_h Y − max_h Ŷ; stepwise_upper (max_h(Y−Ŷ)) retained and renamed. '
                       'Peak exceedance counted on the unified test V=1 population.',
           'configs': {}}
    for ds, seed, bb in CONFIGS:
        r = run(ds, seed, bb)
        rep['configs'][r['config']] = r
        b = r['thresholds']['Q0.99']['methods']
        print(f"== {r['config']} Q0.99: " + ' | '.join(
            f"{m}: S={v['P_S']:.3f} A&S={v['P_A_and_S']:.4f} A|S={v['P_A_given_S']:.4f} "
            f"exceed={v['peak_exceedance_V1']:.3f}"
            for m, v in b.items() if m in ('uncond', 'g1', 'stepwise_upper', 'peak_true')),
            flush=True)
    json.dump(rep, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
