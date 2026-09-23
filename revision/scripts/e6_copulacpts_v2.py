# -*- coding: utf-8 -*-
"""E6-v2：Copula 对照的**成本口径修正**（作者复审 R22-01，本轮唯一必需的真实数据重算）。

问题（已独立核验）：交付版 `scripts/e6_copulacpts.py` 的 `eval_radius` 与 P06 手写统计段
把覆盖放在 V=1 完整路径群体上、把宽度与 IS 放在**逐元素有效掩码**上（含不完整路径里的有效步），
于是同一个"覆盖—成本"工作点的两条轴对应不同群体。合成例：一条完整路径每步宽 2 +
一条不完整路径 11 个有效步每步宽 200 → 交付函数报平均宽 96.6957，V=1 应为 2.0。

本脚本**只改评价群体，不改任何区间**：
  - P00/P01/P02/P08a/P08b 的半径与分位数一律从交付版产物 `p08_copulacpts.json` 读回（冻结）；
  - P06 只在本地重算 g 场（同一 seed、同一 Selection 数据、`early_stopping=False` 的确定性拟合），
    分位数用产物里冻结的 `q_normalized`；
  - 不重新拟合 copula、不调参、不改子采样规模与种子。

因此覆盖数值必须与交付版**逐位一致**（覆盖规则未变），变化的只有宽度/IS 与由它们构成的比较。
脚本把这一点写成硬核验：path_elig / path_hit / 各路径级覆盖若与产物不符即失败退出。

产出：results/phase1_R1/p08_copulacpts_v2.json
      含 ①冻结来源与其 sha256 ②两套群体的宽度/IS（V=1 为正确口径，allvalid 作对照差异展示）
      ③宽度分母 == 12 × 路径分母 的断言 ④合成例复现 ⑤配对块 bootstrap（G1=p5、G2=p95）
用法: python scripts/e6_copulacpts_v2.py
"""
import os
import sys
import json
import time
import hashlib

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
ROOT = os.path.join(PROJ, 'revision_v1')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJ, 'uncalib_v1', 'SRC'))
os.chdir(ROOT)

import conformal as cf                                 # noqa: E402
import r2b_incremental_severity as r2b                 # noqa: E402
import population_v1 as pv                             # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402

ALPHA, H, K = 0.10, 12, 5
N_BOOT, P_RESTART = 2000, 1.0 / 288.0
GROUPS = {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}
OUT = os.path.join(ROOT, 'results', 'phase1_R1', 'p08_copulacpts_v2.json')
SRC = os.path.join(ROOT, 'results', 'phase1_R1', 'p08_copulacpts.json')
CFGKEYS = [('PEMS04', 42, 'PEMS04_s42'), ('PEMS04', 123, 'PEMS04_s123'),
           ('PEMS04', 456, 'PEMS04_s456'), ('PEMS03', 42, 'PEMS03_s42')]
TOL = 1e-9


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load(ds, seed, tag):
    z = np.load(os.path.join(ROOT, 'cache', f'{ds}_s{seed}_{tag}.npz'))
    return {k: z[k] for k in z.files}


def boot_index(n, rng):
    idx = np.empty(n, dtype=np.int64); f = 0
    while f < n:
        st = int(rng.integers(n)); L = int(rng.geometric(P_RESTART))
        b = (st + np.arange(L)) % n; t = min(L, n - f)
        idx[f:f + t] = b[:t]; f += t
    return idx


def synthetic_unit_check():
    """复现审稿包的合成分裂例：一条完整路径每步**宽度** 2；一条不完整路径 11 个有效步每步**宽度** 200。

    注意 half-width 是宽度的一半，故构造时 half 取 1 与 100。
    """
    Hh, N = 12, 1
    pt = np.zeros((2, Hh, N))
    tt = np.full((2, Hh, N), 0.5)
    tt[1, 0, 0] = 0.0                                   # 第二条路径第 1 步无效 ⇒ 不完整
    half = np.zeros((2, Hh, N)); half[0, :, :] = 1.0; half[1, :, :] = 100.0
    lab = np.zeros((2, N), dtype=np.int8)
    v1 = pv.suff_v1(half, pt, tt, lab)
    av = pv.suff_allvalid(half, pt, tt, lab)
    w1 = pv.rat(v1, 'w_sum', 'w_n'); wa = pv.rat(av, 'w_sum', 'w_n')
    return {'note': '完整路径 12 步各宽 2；不完整路径 11 个有效步各宽 200',
            'V1_mean_width': w1, 'V1_width_denominator': float(v1['w_n'].sum()),
            'V1_path_denominator': float(v1['path_elig'].sum()),
            'allvalid_mean_width': wa,
            'allvalid_width_denominator': float(av['w_n'].sum()),
            'allvalid_path_denominator': float(av['path_elig'].sum()),
            'reproduces_delivered_split': abs(wa - 96.69565217391305) < 1e-9,
            'V1_is_correct': abs(w1 - 2.0) < 1e-12}


def halfwidth_fields(ds, seed, frozen, dsel, dcal, dtest):
    """由冻结半径/分位数重建六个程序的半宽场 (n,H,N)。不重新拟合 copula。"""
    pt = dtest['pred'].astype(np.float64)
    dsel_local = dsel
    a, _ = r2b.fit_scale(dsel_local)
    out = {}
    out['P00_path_global'] = np.full(pt.shape, float(frozen['P00_path_global']['radius_uniform']))
    qh = np.asarray(frozen['P01_bonferroni_aH']['radius_per_horizon'], float)
    out['P01_bonferroni_aH'] = np.broadcast_to(qh[None, :, None], pt.shape).copy()
    qa = float(frozen['P02_nh_norm_max']['radius_uniform_normalized'])
    out['P02_nh_norm_max'] = qa * a[None, :, :]
    # P06：重算 g 场（确定性），分位数用冻结值
    g = r2b.g_ranges(ds)
    Xs = r2b.features(dsel_local, g['sel'], a, False)
    ys, vs = r2b.path_risk(dsel_local, a)
    mdl = HistGradientBoostingRegressor(random_state=seed, **r2b.LEARNER).fit(Xs[vs.ravel()], ys)
    n_c, n_t, N = dcal['pred'].shape[0], pt.shape[0], a.shape[1]
    gc_ = np.maximum(mdl.predict(r2b.features(dcal, g['cal'], a, False)), 0.0).reshape(n_c, N)
    gt_ = np.maximum(mdl.predict(r2b.features(dtest, g['test'], a, False)), 0.0).reshape(n_t, N)
    allvc = (dcal['true'] > 0).all(axis=1)
    fl = r2b.G_FLOOR_REL * float(np.median(gc_[allvc]))
    gc_, gt_ = np.maximum(gc_, fl), np.maximum(gt_, fl)
    qg = float(frozen['P06_g0_base']['q_normalized'])
    out['P06_g0_base'] = qg * a[None, :, :] * gt_[:, None, :]
    out['P06_provenance'] = {'q_normalized_frozen': qg, 'g_floor_rel': r2b.G_FLOOR_REL,
                             'learner': dict(r2b.LEARNER, random_state=seed)}
    rad_a = np.asarray(frozen['P08a_copulacpts_native']['radius'], float)
    out['P08a_copulacpts_native'] = np.broadcast_to(rad_a[None, :, None], pt.shape).copy()
    rad_b = np.asarray(frozen['P08b_copulacpts_nh_adapted']['radius'], float)
    out['P08b_copulacpts_nh_adapted'] = rad_b[None, :, None] * a[None, :, :]
    return out, a


def main():
    t_all = time.time()
    src = json.load(open(SRC, encoding='utf-8'))
    out = {
        'script': 'scripts/e6_copulacpts_v2.py',
        'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'purpose': 'R22-01：把 Copula 对照的成本口径统一到 V=1 完整路径群体；区间一律冻结复用',
        'frozen_source': {'path': os.path.relpath(SRC, ROOT).replace('\\', '/'),
                          'sha256': sha(SRC),
                          'note': '半径/分位数全部读回，不重新拟合 copula、不调参、不改子采样与种子'},
        'fix': '交付版 eval_radius 与 P06 统计段的 w_sum/w_n 用逐元素掩码 (tt>0)，'
               '本版全部改用 population_v1.suff_v1（宽度分母=12×V=1 路径数）',
        'acceptance': {'width_denominator_equals_12x_paths': True,
                       'path_counts_and_coverages_unchanged': True},
        'synthetic_check': synthetic_unit_check(),
        'configs': {},
    }
    for ds, seed, key in CFGKEYS:
        t0 = time.time()
        dsel, dcal, dtest = (load(ds, seed, x) for x in ('sel', 'cal', 'test'))
        frozen = src['configs'][key]
        lab = dtest['labels']
        pt, tt = dtest['pred'], dtest['true']
        fields, a = halfwidth_fields(ds, seed, frozen, dsel, dcal, dtest)
        order = ['P00_path_global', 'P01_bonferroni_aH', 'P02_nh_norm_max', 'P06_g0_base',
                 'P08a_copulacpts_native', 'P08b_copulacpts_nh_adapted']
        S, entry = {}, {}
        bad = []
        for m in order:
            s = pv.suff_both(fields[m], pt, tt, lab)
            S[m] = s
            v1, av = s['V1'], s['allvalid']
            old = frozen[m]
            new_cov = pv.rat(v1, 'path_hit', 'path_elig')
            # 硬核验：覆盖与路径计数必须与交付版一致（覆盖规则未变）
            if abs(new_cov - float(old['path_cov'])) > 1e-12:
                bad.append((m, 'path_cov changed', new_cov, float(old['path_cov'])))
            if abs(float(v1['path_elig'].sum()) - float(old['n_paths'])) > 0.5:
                bad.append((m, 'n_paths changed', float(v1['path_elig'].sum()), old['n_paths']))
            for g in GROUPS.values():
                gc_new = pv.rat(v1, f'{g}_hit', f'{g}_elig')
                if abs(gc_new - float(old['group_path_cov'][g])) > 1e-12:
                    bad.append((m, f'{g} cov changed', gc_new, float(old['group_path_cov'][g])))
            if v1['w_n'].sum() != 12 * v1['path_elig'].sum():
                bad.append((m, 'width denominator != 12 x path denominator'))
            entry[m] = {
                'V1': {'path_cov': new_cov,
                       'mean_width': pv.rat(v1, 'w_sum', 'w_n'),
                       'IS': pv.rat(v1, 'is_sum', 'w_n'),
                       'n_paths': float(v1['path_elig'].sum()),
                       'width_denominator': float(v1['w_n'].sum()),
                       'group_path_cov': {g: pv.rat(v1, f'{g}_hit', f'{g}_elig')
                                           for g in GROUPS.values()}},
                'allvalid_as_delivered': {'mean_width': pv.rat(av, 'w_sum', 'w_n'),
                                          'IS': pv.rat(av, 'is_sum', 'w_n'),
                                          'width_denominator': float(av['w_n'].sum())},
                'delivered_artifact': {'path_cov': float(old['path_cov']),
                                       'mean_width': float(old['mean_width']),
                                       'IS': float(old['IS']),
                                       'n_paths': float(old['n_paths'])},
                'delta_width_pct_vs_delivered': 100.0 * (pv.rat(v1, 'w_sum', 'w_n')
                                                          / float(old['mean_width']) - 1.0),
                'delta_IS_pct_vs_delivered': 100.0 * (pv.rat(v1, 'is_sum', 'w_n')
                                                      / float(old['IS']) - 1.0)}
        entry['P06_provenance'] = fields['P06_provenance']
        # 配对块 bootstrap（索引跨方法共享），对 P02 与 P00 两个参照
        rng = np.random.default_rng(7)
        n = pt.shape[0]
        boots = []
        for _ in range(N_BOOT):
            boots.append(boot_index(n, rng))
        contr = {}
        for m, ref in (('P08a_copulacpts_native', 'P02_nh_norm_max'),
                       ('P08b_copulacpts_nh_adapted', 'P02_nh_norm_max'),
                       ('P08a_copulacpts_native', 'P00_path_global'),
                       ('P06_g0_base', 'P08a_copulacpts_native')):
            dc, wr = [], []
            for bi in boots:
                Sm = {k: v[bi] for k, v in S[m]['V1'].items()}
                Sr = {k: v[bi] for k, v in S[ref]['V1'].items()}
                dc.append(pv.rat(Sm, 'path_hit', 'path_elig') - pv.rat(Sr, 'path_hit', 'path_elig'))
                wr.append(pv.rat(Sm, 'w_sum', 'w_n') / pv.rat(Sr, 'w_sum', 'w_n'))
            dc, wr = np.array(dc), np.array(wr)
            contr[f'{m}__vs__{ref}'] = {
                'd_path_cov_mean': float(dc.mean()),
                'G1_one_sided95_lower_p5': float(np.percentile(dc, 5.0)),
                'G1_pass': bool(np.percentile(dc, 5.0) >= -0.01),
                'width_ratio_mean': float(wr.mean()),
                'G2_one_sided95_upper_p95': float(np.percentile(wr, 95.0)),
                'G2_pass': bool(np.percentile(wr, 95.0) <= 0.95),
                'width_ratio_ci95': [float(np.percentile(wr, 2.5)), float(np.percentile(wr, 97.5))],
                'achieved_cov_candidate': entry[m]['V1']['path_cov'],
                'achieved_cov_reference': entry[ref]['V1']['path_cov']}
        entry['contrasts'] = contr
        entry['verification'] = {'coverage_and_path_counts_identical_to_delivered': not bad,
                                 'issues': bad}
        entry['runtime_sec'] = round(time.time() - t0, 1)
        out['configs'][key] = entry
        stat = 'FAIL' if bad else 'OK'
        print(f'== {key} [{stat}] ({entry["runtime_sec"]}s)', flush=True)
        for b in bad:
            print('    ISSUE', b, flush=True)
        for m in order:
            e = entry[m]
            print(f'   {m:26s} cov={e["V1"]["path_cov"]:.4f} '
                  f'widthV1={e["V1"]["mean_width"]:7.1f} (delivered {e["allvalid_as_delivered"]["mean_width"]:7.1f}'
                  f', {e["delta_width_pct_vs_delivered"]:+.2f}%) '
                  f'ISV1={e["V1"]["IS"]:7.1f} ({e["delta_IS_pct_vs_delivered"]:+.2f}%)', flush=True)
        for k, v in contr.items():
            print(f'   {k:46s} wr p95={v["G2_one_sided95_upper_p95"]:.4f}'
                  f'{" PASS" if v["G2_pass"] else " FAIL"} cov {v["achieved_cov_candidate"]:.4f}'
                  f' vs {v["achieved_cov_reference"]:.4f}', flush=True)
    all_bad = [b for c in out['configs'].values() for b in c['verification']['issues']]
    out['acceptance']['path_counts_and_coverages_unchanged'] = not all_bad
    out['acceptance']['synthetic_reproduces_delivered_split'] = \
        out['synthetic_check']['reproduces_delivered_split']
    out['summary'] = {}
    for m in ['P00_path_global', 'P01_bonferroni_aH', 'P02_nh_norm_max', 'P06_g0_base',
              'P08a_copulacpts_native', 'P08b_copulacpts_nh_adapted']:
        ws = [out['configs'][k][m]['V1']['mean_width'] for _, _, k in CFGKEYS]
        wv = [out['configs'][k][m]['allvalid_as_delivered']['mean_width'] for _, _, k in CFGKEYS]
        cov = [out['configs'][k][m]['V1']['path_cov'] for _, _, k in CFGKEYS]
        out['summary'][m] = {'mean_width_V1': float(np.mean(ws)),
                             'mean_width_as_delivered': float(np.mean(wv)),
                             'mean_delta_width_pct': float(np.mean(
                                 [100 * (o - d) / d for o, d in zip(ws, wv)])),
                             'mean_path_cov': float(np.mean(cov))}
    out['runtime_sec'] = round(time.time() - t_all, 1)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nsaved ->', OUT, '| runtime', out['runtime_sec'], 's')
    print('acceptance:', json.dumps(out['acceptance'], ensure_ascii=False))
    if all_bad:
        raise SystemExit('验收失败：覆盖/路径计数与交付版不一致，先查实现')


if __name__ == '__main__':
    main()
