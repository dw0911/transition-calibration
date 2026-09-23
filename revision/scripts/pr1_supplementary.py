# -*- coding: utf-8 -*-
"""第一轮论文审查（PR1）的补充统计：A3 实验身份、A4 效应量、B1 非负支持集、B2 前瞻路径群体。

边界（审查 02 表 A/B 段 + 用户"必须完成：主要使用已有材料"）：
  - 不训练、不调参、不重新选择分组或阈值。q 场由 scripts/e7_metrics_v1.build_q 用冻结的
    cache 导出与冻结常量**重新求值**，并逐配置与已登记产物对账：路径计数、覆盖、宽度、
    bootstrap p5/p95 与 G1/G2 结论必须一致，不一致即 assert 中止。
  - results/phase2_E7/v1/*.json 与 SUMMARY.json 只读；本脚本不写回任何已登记产物。
  - B1 是成本定义的敏感性（区间与 [0,+∞) 相交），不是新算法：命中计数必须逐位不变。
  - B2 是本阶段的补充诊断，正文必须标注其非预注册身份。

输出: results/phase3_PR1/pr1_supplementary.json
"""
import os
import sys
import json
import time

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(os.path.join(_PKG, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

import population_v1 as pv                          # noqa: E402
import r2b_incremental_severity as r2b              # noqa: E402
import e7_metrics_v1 as v1                          # noqa: E402

V1DIR = os.path.join(ROOT, 'results', 'phase2_E7', 'v1')
OUTDIR = os.path.join(ROOT, 'results', 'phase3_PR1')
OUT = os.path.join(OUTDIR, 'pr1_supplementary.json')
CONFIGS = v1.CONFIGS
PAIRS = v1.PAIRS
METHODS = ['uncond', 'sev', 'flow', 'pmag', 'nh_elem', 'nh_path',
           'path_global', 'bonferroni_aH', 'g0', 'g1']
B2_METHODS = ['uncond', 'path_global', 'nh_path', 'g0', 'g1']
TOL_COV = 1e-12
TOL_W = 1e-9


# ---------------------------------------------------------------- population
def seg_counts(tag, d):
    """一个数据段的人群计数（窗口、节点、V=1 路径、非正目标）。

    ``labels``（事后事件组）只在 test 导出中存在，因此其它段记 None。
    """
    tt = d['true'].astype(np.float64)
    vt = tt > 0
    allv = vt.all(axis=1)
    n, H, N = tt.shape
    out = {'tag': tag, 'n_windows': int(n), 'H': int(H), 'n_nodes': int(N),
           'n_node_paths': int(n * N), 'v1_paths': int(allv.sum()),
           'v1_rate': float(allv.mean()),
           'paths_with_any_nonpositive': int((~allv).sum()),
           'nonpositive_elements': int((~vt).sum()),
           'nonpositive_element_rate': float(1.0 - vt.mean()),
           'paths_all_nonpositive': int((~vt).all(axis=1).sum()),
           'event_group_path_counts': None}
    if 'labels' in d:
        lab = d['labels']
        out['event_group_path_counts'] = {g: int(((lab == tid) & allv).sum())
                                          for tid, g in pv.GROUPS.items()}
    return out


def point_quality(d):
    """同一有效群体上的点预测基本误差（流量单位）。"""
    p, t = d['pred'].astype(np.float64), d['true'].astype(np.float64)
    vt, allv = t > 0, (t > 0).all(axis=1)
    e = np.abs(p - t)
    m3 = np.broadcast_to(allv[:, None, :], e.shape)
    return {'MAE_valid_elements': float(e[vt].mean()),
            'RMSE_valid_elements': float(np.sqrt((e[vt] ** 2).mean())),
            'MAE_V1_population': float(e[m3].mean()),
            'n_valid_elements': int(vt.sum()),
            'median_AE_valid_elements': float(np.median(e[vt]))}


def prospective_masks(dtest, dsel):
    """固定分组：输入转变标记 × Selection 三分位（阈值取选择段全有效路径的分位）。"""
    hi = (dtest['labels'] == 1) | (dtest['labels'] == 3)
    sel_ok = (dsel['true'] > 0).all(axis=1)
    groups = {}
    for gname, key in (('mag', 'pred_mag'), ('flow', 'flow_in')):
        cuts = np.quantile(dsel[key][sel_ok], [1 / 3, 2 / 3])
        tier = np.digitize(dtest[key], cuts)
        cells = {f'hi{int(hb)}_t{tb}': (hi == hb) & (tier == tb)
                 for hb in (False, True) for tb in (0, 1, 2)}
        groups[gname] = {'cut_points_selection': [float(c) for c in cuts], 'cells': cells}
    return groups


# ------------------------------------------------------------- method fields
def node_fields(qm, pt64, tt64, allv):
    """把一个程序的 (n,H,N) 场折叠到 (n,N) 节点级统计量，避免多次大数组分配。"""
    half = np.broadcast_to(np.asarray(qm, np.float64), pt64.shape)
    lo, hi = pt64 - half, pt64 + half
    cov = (tt64 >= lo) & (tt64 <= hi)                      # 原对称区间
    covc = (tt64 >= np.maximum(lo, 0.0)) & (tt64 <= hi)   # C ∩ [0,∞)
    w = 2.0 * half
    wc = np.maximum(0.0, hi - np.maximum(0.0, lo))
    H = pt64.shape[1]
    out = {'path_hit': cov.all(axis=1), 'path_hit_clip': covc.all(axis=1),
           'elem_hit': cov.sum(axis=1).astype(np.int64),
           'elem_hit_clip': covc.sum(axis=1).astype(np.int64),
           'w_mean': w.mean(axis=1), 'wc_mean': wc.mean(axis=1),
           'n_lo_neg': (lo < 0).sum(axis=1).astype(np.int64),
           'n_hi_nonpos': (hi <= 0).sum(axis=1).astype(np.int64),
           'w_sum': w.sum(axis=1), 'wc_sum': wc.sum(axis=1),
           'H': H}
    del half, lo, hi, cov, covc, w, wc
    return out


def clip_stats(f, allv):
    H = f['H']
    sel = allv
    n_paths = int(sel.sum())
    den = n_paths * H
    w_sum, wc_sum = float(f['w_sum'][sel].sum()), float(f['wc_sum'][sel].sum())
    return {'v1_paths': n_paths, 'v1_elements': den,
            'mean_width_original': w_sum / den,
            'mean_width_clipped': wc_sum / den,
            'clipped_over_original': wc_sum / w_sum,
            'width_reduction_pct': 100.0 * (1.0 - wc_sum / w_sum),
            'frac_lower_below_zero': float(f['n_lo_neg'][sel].sum()) / den,
            'frac_upper_nonpositive': float(f['n_hi_nonpos'][sel].sum()) / den,
            'element_hits_original': int(f['elem_hit'][sel].sum()),
            'element_hits_clipped': int(f['elem_hit_clip'][sel].sum()),
            'path_hits_original': int(f['path_hit'][sel].sum()),
            'path_hits_clipped': int(f['path_hit_clip'][sel].sum())}


def cell_stats(f, cell_mask):
    n = int(cell_mask.sum())
    if n == 0:
        return {'n_v1_paths': 0}
    H = f['H']
    return {'n_v1_paths': n,
            'path_cov': float(f['path_hit'][cell_mask].mean()),
            'path_hits': int(f['path_hit'][cell_mask].sum()),
            'mean_width': float(f['w_sum'][cell_mask].sum() / (n * H)),
            'mean_width_clipped': float(f['wc_sum'][cell_mask].sum() / (n * H)),
            'frac_lower_below_zero': float(f['n_lo_neg'][cell_mask].sum()) / (n * H)}


# ------------------------------------------------------------------------- run
def per_config(ds, seed, bb):
    t0 = time.time()
    name = f'{ds}_{bb}_s{seed}'
    dsel, dcal, dtest = (v1.load(ds, seed, 'sel', bb), v1.load(ds, seed, 'cal', bb),
                         v1.load(ds, seed, 'test', bb))
    dsel['dataset'] = ds
    stored = json.load(open(os.path.join(V1DIR, f'{name}.json'), encoding='utf-8'))
    pt, tt, lab = dtest['pred'], dtest['true'], dtest['labels']
    pt64, tt64 = pt.astype(np.float64), tt.astype(np.float64)
    allv = (tt64 > 0).all(axis=1)
    q, a, n_fb = v1.build_q(dsel, dcal, dtest, seed)

    S = {m: pv.suff_both(q[m], pt, tt, lab) for m in METHODS}
    met = {m: pv.metrics(S[m]['V1']) for m in METHODS}

    # ---- 对账 1：覆盖 / 宽度 / 路径计数与已登记产物一致 ---------------------------
    ver, worst = {}, 0.0
    for m in METHODS:
        s = stored['metrics'][m]['V1']
        d = {'path_cov': abs(met[m]['path_cov'] - s['path_cov']),
             'pointwise_same_pop': abs(met[m]['pointwise_same_pop'] - s['pointwise_same_pop']),
             'mean_width': abs(met[m]['mean_width'] - s['mean_width']),
             'n_paths_diff': abs(met[m]['n_paths'] - s['n_paths'])}
        ver[m] = d
        assert d['n_paths_diff'] == 0, f'{name}/{m}: 路径计数不同'
        assert d['path_cov'] <= TOL_COV and d['pointwise_same_pop'] <= TOL_COV, \
            f'{name}/{m}: 覆盖不一致 {d}'
        assert d['mean_width'] <= TOL_W, f'{name}/{m}: 宽度不一致 {d}'
        worst = max(worst, d['path_cov'], d['pointwise_same_pop'], d['mean_width'])

    # ---- A3：身份、人群计数、点预测质量 ------------------------------------------
    ident = {'config': name, 'dataset': ds, 'backbone': bb, 'seed': seed,
             'segments': [seg_counts(t, d) for t, d in
                          (('selection', dsel), ('calibration', dcal), ('test', dtest))],
             'point_quality': {'selection': point_quality(dsel),
                               'calibration': point_quality(dcal),
                               'test': point_quality(dtest)},
             'a_fallback_cells': int(n_fb),
             'zero_convention': {'valid_rule': 'true > 0',
                                 'test_elements_nonpositive': int((tt64 <= 0).sum()),
                                 'test_elements_exactly_zero': int((tt64 == 0).sum()),
                                 'test_elements_negative': int((tt64 < 0).sum())}}
    prov = {}
    suf = '' if bb == 'stae' else '_stid'
    em = os.path.join(ROOT, 'cache', f'{ds}{suf}_s{seed}_export_meta.json')
    if os.path.exists(em):
        prov['export_meta'] = json.load(open(em, encoding='utf-8'))
    for cand in (os.path.join(ROOT, 'checkpoints', f'{ds}{suf}_e7_4split_seed{seed}',
                              'train_meta.json'),
                 os.path.join(PROJ, 'uncalib_v1', 'EXPERIMENTS', 'UC-G3', 'checkpoints',
                              f'{ds}_r3_4split_seed{seed}', 'train_meta.json')):
        if os.path.exists(cand):
            prov['train_meta'] = {'path': os.path.relpath(cand, PROJ).replace('\\', '/'),
                                  **json.load(open(cand, encoding='utf-8'))}
            break
    prov['exported_by'] = ('scripts/e7_export.py' if 'export_meta' in prov
                           else 'scripts/export_preds.py')
    ident['provenance'] = prov

    # ---- B1 + B2：逐程序一遍折叠，命中计数不变性在循环内断言 ---------------------
    pg = prospective_masks(dtest, dsel)
    clip, pro = {}, {g: {'cut_points_selection': gd['cut_points_selection'], 'cells': {}}
                     for g, gd in pg.items()}
    for m in METHODS:
        f = node_fields(q[m], pt64, tt64, allv)
        cs = clip_stats(f, allv)
        assert cs['element_hits_original'] == cs['element_hits_clipped'], \
            f'{name}/{m}: 非负相交改变元素命中数'
        assert cs['path_hits_original'] == cs['path_hits_clipped'], \
            f'{name}/{m}: 非负相交改变路径命中数'
        assert cs['path_hits_original'] / max(cs['v1_paths'], 1) == met[m]['path_cov'], \
            f'{name}/{m}: 折叠后的路径覆盖与 suff 口径不同'
        clip[m] = dict(cs, element_hits_unchanged=True, path_hits_unchanged=True)
        if m in B2_METHODS:
            for gname, gd in pg.items():
                for cname, mask in gd['cells'].items():
                    cell_mask = mask & allv
                    st = cell_stats(f, cell_mask)
                    st['share_of_v1_paths'] = st['n_v1_paths'] / max(int(allv.sum()), 1)
                    pro[gname]['cells'].setdefault(cname, {})[m] = st
        del f
    for gname in pro:                                  # 按群体的路径数与程序无关
        for cname, c in pro[gname]['cells'].items():
            c['n_v1_paths_ref'] = c['uncond']['n_v1_paths']

    # ---- A4：效应量 + 同一冻结 bootstrap 分布的额外分位数 ------------------------
    rng = np.random.default_rng(20260921)
    n = pt.shape[0]
    eff, bver = {}, {}
    for m, r in PAIRS:
        Sm, Sr = S[m]['V1'], S[r]['V1']
        dc, wr = [], []
        for _ in range(v1.N_BOOT):
            bi = r2b.stationary_index(n, rng)
            mb = {k: vv[bi] for k, vv in Sm.items()}
            rb = {k: vv[bi] for k, vv in Sr.items()}
            dc.append(pv.rat(mb, 'path_hit', 'path_elig') - pv.rat(rb, 'path_hit', 'path_elig'))
            wr.append(pv.rat(mb, 'w_sum', 'w_n') / pv.rat(rb, 'w_sum', 'w_n'))
        dc, wr = np.array(dc), np.array(wr)
        key = f'{m}__vs__{r}'
        st = stored['contrasts'][key]
        eff[key] = {'candidate': m, 'reference': r,
                    'delta_coverage_point_estimate': met[m]['path_cov'] - met[r]['path_cov'],
                    'width_ratio_point_estimate': met[m]['mean_width'] / met[r]['mean_width'],
                    'coverage_candidate': met[m]['path_cov'], 'coverage_reference': met[r]['path_cov'],
                    'width_candidate': met[m]['mean_width'], 'width_reference': met[r]['mean_width'],
                    'boot_mean_delta': float(dc.mean()), 'boot_median_delta': float(np.median(dc)),
                    'delta_CI95_two_sided': [float(np.percentile(dc, 2.5)),
                                             float(np.percentile(dc, 97.5))],
                    'delta_one_sided_lower_p5': float(np.percentile(dc, 5.0)),
                    'G1_pass': bool(np.percentile(dc, 5.0) >= -0.01),
                    'boot_mean_ratio': float(wr.mean()),
                    'ratio_CI95_two_sided': [float(np.percentile(wr, 2.5)),
                                             float(np.percentile(wr, 97.5))],
                    'ratio_one_sided_upper_p95': float(np.percentile(wr, 95.0)),
                    'G2_pass': bool(np.percentile(wr, 95.0) <= 0.95)}
        bver[key] = {'delta_p5_abs': abs(eff[key]['delta_one_sided_lower_p5']
                                         - st['G1_one_sided95_lower_p5']),
                     'ratio_p95_abs': abs(eff[key]['ratio_one_sided_upper_p95']
                                          - st['G2_one_sided95_upper_p95']),
                     'G1_same': eff[key]['G1_pass'] == st['G1_pass'],
                     'G2_same': eff[key]['G2_pass'] == st['G2_pass']}
        assert bver[key]['G1_same'] and bver[key]['G2_same'], f'{name}/{key}: 判据结论不同'
        assert bver[key]['delta_p5_abs'] < 1e-9 and bver[key]['ratio_p95_abs'] < 1e-9, \
            f'{name}/{key}: bootstrap 分位数与已登记产物不同 {bver[key]}'

    return {'identity': ident, 'support_clip': clip, 'prospective_paths': pro,
            'effect_sizes': eff,
            'reproduction_check': {'per_method_abs_diff': ver, 'worst_coverage_or_width': worst,
                                   'bootstrap_percentile_diff': bver,
                                   'verdict': 'IDENTICAL TO REGISTERED ARTIFACT'},
            'runtime_sec': round(time.time() - t0, 1)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = {'task': 'PR1 supplementary statistics (A3 identity, A4 effect sizes, '
                   'B1 non-negative-support width sensitivity, B2 prospective-group path '
                   'coverage)',
           'frozen_inputs': {'predictions': 'revision_v1/cache/*.npz',
                             'calibrators': 'scripts/e7_metrics_v1.build_q (unmodified)',
                             'bootstrap': 'rng 20260921, 2000 replicates, p=1/288, shared '
                                          'indices, PAIRS order as in e7_metrics_v1.py'},
           'no_reselection': 'no method, threshold, grouping, feature or model was re-selected; '
                             'every q field is re-evaluated from frozen constants and verified '
                             'against results/phase2_E7/v1/*.json before any new statistic is taken',
           'configs': {}}
    for ds, seed, bb in CONFIGS:
        rec = per_config(ds, seed, bb)
        out['configs'][f'{ds}_{bb}_s{seed}'] = rec
        i = rec['identity']
        print(f'== {ds}_{bb}_s{seed} ({rec["runtime_sec"]}s) V1rate(test)='
              f'{i["segments"][2]["v1_rate"]:.4f} MAE_test={i["point_quality"]["test"]["MAE_valid_elements"]:.3f}',
              flush=True)
        for m in METHODS:
            c = rec['support_clip'][m]
            print(f'   {m:14s} W={c["mean_width_original"]:8.2f} W+={c["mean_width_clipped"]:8.2f} '
                  f'({100 * c["clipped_over_original"]:6.2f}%)  lo<0={c["frac_lower_below_zero"]:.4f} '
                  f'hi<=0={c["frac_upper_nonpositive"]:.5f}', flush=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
