# -*- coding: utf-8 -*-
"""V=1 统一口径重算（作者校审任务 2）+ 协议化验收判据（任务 C-F）。

与 e7_metrics.py 的关系：不覆盖旧产物。本脚本写到 results/phase2_E7/v1/，
旧目录保持可复现（DEV-E7-4 的引用链仍指向旧目录的 block_A/block_B）。

口径（protocol.yaml :: gates.population / output_requirement）：
  - 主表 path_cov / pointwise_same_pop / mean_width / IS 全部在 **V=1 完整路径群体**；
    全有效元素口径作为辅助列保留（population_v1.suff_both）。
  - 验收判据分开输出：G1 = Δpath_cov 配对块 bootstrap 的 p5 ≥ −0.01；
    G2 = 宽度比 bootstrap 的 **p95** ≤ 0.95；实际覆盖率单独描述；
    **不再**把"两方法覆盖 ≥0.89"并入合取。
  - bootstrap 索引跨方法共享、每 config 新建 rng(20260921)，与旧链一致。

方法族：uncond / sev / flow / pmag / nh_elem（element 级）+ path_global / bonferroni_aH
（路径级简单参照）+ g0 / g1（学习型可观测风险）。

输出: results/phase2_E7/v1/<config>.json 与 v1/SUMMARY.json（数据集×骨干子组均值）。
"""
import os
import sys
import json
import time

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
import population_v1 as pv                            # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402

ALPHA, K, H = 0.10, 5, 12
N_BOOT = 2000
OUT = os.path.join(ROOT, 'results', 'phase2_E7', 'v1')
CONFIGS = [('PEMS04', 42, 'stae'), ('PEMS04', 123, 'stae'), ('PEMS04', 456, 'stae'),
           ('PEMS03', 42, 'stae'), ('PEMS03', 123, 'stae'), ('PEMS03', 456, 'stae'),
           ('PEMS08', 42, 'stae'), ('PEMS08', 123, 'stae'), ('PEMS08', 456, 'stae'),
           ('PEMS08', 42, 'stid'), ('PEMS08', 123, 'stid'), ('PEMS08', 456, 'stid')]
PAIRS = [('g1', 'g0'), ('g1', 'nh_elem'), ('nh_elem', 'uncond'),
         ('bonferroni_aH', 'path_global'), ('g1', 'path_global'),
         ('nh_path', 'path_global'), ('g1', 'nh_path')]


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


def build_q(dsel, dcal, dtest, seed):
    pc, tc = dcal['pred'], dcal['true']
    pt, tt = dtest['pred'], dtest['true']
    vc = tc > 0
    Rc = np.abs(pc - tc).astype(np.float64)
    a, n_fb = r2b.fit_scale(dsel)
    allv_c = vc.all(axis=1)
    q = {}
    q['uncond'] = np.full(pt.shape, cf.unconditional(pc, tc, pt, tt, dtest['labels'], ALPHA,
                                                     valid_val=vc, valid_test=tt > 0)[1])
    for nm, ck in (('sev', 'r_in_max'), ('flow', 'flow_in'), ('pmag', 'pred_mag')):
        _, m = cf.mondrian_hard(pc, tc, pt, tt, dtest['labels'],
                                dcal[ck].astype(np.float64), dtest[ck].astype(np.float64),
                                K=K, alpha=ALPHA, valid_val=vc, valid_test=tt > 0)
        e, qb = np.asarray(m['edges']), np.asarray(m['q_bins'])
        b = np.clip(np.digitize(cf._broadcast_cond(dtest[ck].astype(np.float64), pt.shape),
                                e[1:-1]), 0, K - 1)
        q[nm] = qb[b]
    q['nh_elem'] = cf.conformal_quantile((Rc / a[None, :, :])[vc], ALPHA) * a[None, :, :]
    q['nh_path'] = cf.conformal_quantile((Rc / a[None, :, :]).max(axis=1)[allv_c], ALPHA)         * a[None, :, :]
    q['path_global'] = np.full(pt.shape, cf.conformal_quantile(Rc.max(axis=1)[allv_c], ALPHA))
    qh = np.array([cf.conformal_quantile(Rc[:, h, :][vc[:, h, :]], ALPHA / H) for h in range(H)])
    q['bonferroni_aH'] = np.broadcast_to(qh[None, :, None], pt.shape).copy()
    # g0/g1
    g = g_ranges(dsel['dataset']) if 'dataset' in dsel else None
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
    return q, a, n_fb


def run(ds, seed, bb):
    t0 = time.time()
    name = f'{ds}_{bb}_s{seed}'
    dsel, dcal, dtest = load(ds, seed, 'sel', bb), load(ds, seed, 'cal', bb), \
        load(ds, seed, 'test', bb)
    dsel['dataset'] = ds
    pt, tt, lab = dtest['pred'], dtest['true'], dtest['labels']
    q, a, n_fb = build_q(dsel, dcal, dtest, seed)
    S = {m: pv.suff_both(qq, pt, tt, lab) for m, qq in q.items()}
    metrics = {m: {'V1': pv.metrics(s['V1']), 'allvalid': pv.metrics(s['allvalid'])}
               for m, s in S.items()}
    rng = np.random.default_rng(20260921)
    n = pt.shape[0]
    contr = {}
    for m, r in PAIRS:
        Sm, Sr = S[m]['V1'], S[r]['V1']
        dc, wr = [], []
        for _ in range(N_BOOT):
            bi = r2b.stationary_index(n, rng)
            mb = {k: v[bi] for k, v in Sm.items()}
            rb = {k: v[bi] for k, v in Sr.items()}
            dc.append(pv.rat(mb, 'path_hit', 'path_elig') - pv.rat(rb, 'path_hit', 'path_elig'))
            wr.append(pv.rat(mb, 'w_sum', 'w_n') / pv.rat(rb, 'w_sum', 'w_n'))
        dc, wr = np.array(dc), np.array(wr)
        cov_m = metrics[m]['V1']['path_cov']
        cov_r = metrics[r]['V1']['path_cov']
        contr[f'{m}__vs__{r}'] = {
            'd_path_cov_mean': float(dc.mean()),
            'G1_one_sided95_lower_p5': float(np.percentile(dc, 5.0)),
            'G1_pass': bool(np.percentile(dc, 5.0) >= -0.01),
            'width_ratio_mean': float(wr.mean()),
            'G2_one_sided95_upper_p95': float(np.percentile(wr, 95.0)),
            'G2_pass': bool(np.percentile(wr, 95.0) <= 0.95),
            'width_ratio_CI95': [float(np.percentile(wr, 2.5)), float(np.percentile(wr, 97.5))],
            'achieved_cov_candidate': cov_m, 'achieved_cov_reference': cov_r,
            'gate_pass_protocol': bool(np.percentile(dc, 5.0) >= -0.01
                                       and np.percentile(wr, 95.0) <= 0.95)}
    rec = {'config': name, 'status': 'COMPLETED', 'population': 'V=1 for main metrics; '
           'allvalid kept as auxiliary', 'n_boot': N_BOOT, 'a_fallback_cells': n_fb,
           'metrics': metrics, 'contrasts': contr,
           'runtime_sec': round(time.time() - t0, 1)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, name + '.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    m = metrics
    print(f'== {name} ({rec["runtime_sec"]}s) paths={m["uncond"]["V1"]["n_paths"]:.0f}', flush=True)
    for nm in ('uncond', 'path_global', 'bonferroni_aH', 'nh_elem', 'nh_path', 'g0', 'g1'):
        v = m[nm]['V1']
        print(f'   {nm:14s} pathCov={v["path_cov"]:.4f} widthV1={v["mean_width"]:7.1f} '
              f'widthAll={m[nm]["allvalid"]["mean_width"]:7.1f} '
              f'TypeC={v["group_path_cov"]["TypeC"]:.3f} TypeB={v["group_path_cov"]["TypeB"]:.3f}',
              flush=True)
    for k, v in contr.items():
        print(f'   {k:34s} G1 p5={v["G1_one_sided95_lower_p5"]:+.4f}({"P" if v["G1_pass"] else "F"}) '
              f'G2 p95={v["G2_one_sided95_upper_p95"]:.4f}({"P" if v["G2_pass"] else "F"}) '
              f'cov {v["achieved_cov_candidate"]:.4f}/{v["achieved_cov_reference"]:.4f}', flush=True)
    return rec


def summarize(recs):
    groups = {}
    for r in recs:
        ds, bb = r['config'].split('_')[0], ('stid' if 'stid' in r['config'] else 'stae')
        groups.setdefault(f'{ds}/{bb}', []).append(r)
    out = {'groups': {},
           'ratio_definition': 'mean of per-configuration ratios (equal weight over seeds), '
                               'as declared in Appendix "Bootstrap Settings"; the pooled '
                               'ratio-of-means is kept only as a diagnostic column'}
    for gname, rs in sorted(groups.items()):
        m = lambda meth, f: float(np.mean([f(r['metrics'][meth]['V1']) for r in rs]))
        e = lambda meth, f: float(np.mean([f(r['metrics'][meth]['allvalid']) for r in rs]))

        def mean_ratio(meth, ref, pop):
            """R23-F2：先在每个配置内求 W_m/W_ref，再等权平均（协议声明的汇总层次）。"""
            return float(np.mean([r['metrics'][meth][pop]['mean_width']
                                  / r['metrics'][ref][pop]['mean_width'] for r in rs]))

        def pooled(meth, ref, pop):
            """平均的比值（旧实现）；只作为诊断列保留，不进正文。"""
            w = lambda k: float(np.mean([r['metrics'][k][pop]['mean_width'] for r in rs]))
            return w(meth) / w(ref)

        wv1 = {meth: mean_ratio(meth, 'uncond', 'V1')
               for meth in ('path_global', 'bonferroni_aH', 'nh_elem', 'nh_path', 'g0', 'g1')}
        wall = {meth: mean_ratio(meth, 'uncond', 'allvalid') for meth in ('g1',)}
        wg1g0 = mean_ratio('g1', 'g0', 'V1')
        # 局部断言：两种汇总方式必须仍在"不影响任何结论"的量级内（本轮实测最大 1.9e-4）
        for _k, _v in wv1.items():
            assert abs(_v - pooled(_k, 'uncond', 'V1')) < 1e-3, \
                f'{gname}/V1/{_k}: 两种汇总定义偏离 {abs(_v - pooled(_k, "uncond", "V1")):.2e}'
        for _k, _v in wall.items():
            assert abs(_v - pooled(_k, 'uncond', 'allvalid')) < 1e-3, \
                f'{gname}/allvalid/{_k}: 两种汇总定义偏离'
        assert abs(wg1g0 - pooled('g1', 'g0', 'V1')) < 1e-3, f'{gname}/g1_vs_g0: 汇总定义偏离'
        out['groups'][gname] = {
            'n': len(rs),
            'achieved_path_cov': {meth: m(meth, lambda x: x['path_cov'])
                                  for meth in ('uncond', 'path_global', 'bonferroni_aH',
                                               'nh_elem', 'nh_path', 'g0', 'g1')},
            'width_V1_ratio_vs_uncond': wv1,
            'width_allvalid_ratio_vs_uncond': wall,
            'pooled_ratio_vs_uncond_diag': {meth: pooled(meth, 'uncond', 'V1') for meth in wv1},
            'typeC_realloc_element': m('sev', lambda x: x['group_path_cov']['TypeC'])
            - m('uncond', lambda x: x['group_path_cov']['TypeC']),
            'typeB_realloc_element': m('sev', lambda x: x['group_path_cov']['TypeB'])
            - m('uncond', lambda x: x['group_path_cov']['TypeB']),
            'typeC_incr_matched_g1_minus_g0': m('g1', lambda x: x['group_path_cov']['TypeC'])
            - m('g0', lambda x: x['group_path_cov']['TypeC']),
            # R22-02：匹配模型在 TypeB 上的增量（摘要需要与规则更换效应分开引用）
            'typeB_incr_matched_g1_minus_g0': m('g1', lambda x: x['group_path_cov']['TypeB'])
            - m('g0', lambda x: x['group_path_cov']['TypeB']),
            'g1_vs_g0_width_ratio_V1': wg1g0}
        gates = {}
        for key in [k for k in rs[0]['contrasts']]:
            gates[key] = {'G1_pass_frac': float(np.mean([r['contrasts'][key]['G1_pass']
                                                         for r in rs])),
                          'G2_pass_frac': float(np.mean([r['contrasts'][key]['G2_pass']
                                                         for r in rs])),
                          'gate_pass_frac': float(np.mean([r['contrasts'][key]['gate_pass_protocol']
                                                           for r in rs]))}
        out['groups'][gname]['gate_pass_fraction'] = gates
    json.dump(out, open(os.path.join(OUT, 'SUMMARY.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:2500])
    print('saved ->', os.path.join(OUT, 'SUMMARY.json'))


def main():
    if '--summarize-only' in sys.argv:
        # 只重聚合，不重跑任何分位数/bootstrap：读回 12 个已登记 config JSON 后重建 SUMMARY.json。
        # 用于新增派生字段（如 Bonferroni 实际覆盖）时保证逐位可复现。
        recs = []
        for ds, seed, bb in CONFIGS:
            p = os.path.join(OUT, f'{ds}_{bb}_s{seed}.json')
            if not os.path.exists(p):
                raise SystemExit(f'缺少已登记 config 产物：{p}')
            recs.append(json.load(open(p, encoding='utf-8')))
        summarize(recs)
        return
    recs = []
    for ds, seed, bb in CONFIGS:
        recs.append(run(ds, seed, bb))
    summarize(recs)


if __name__ == '__main__':
    main()
