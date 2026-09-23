# -*- coding: utf-8 -*-
"""D3：事后事件分组的覆盖落差是否"完美模型也会出现"？以及 —— 关键区分 ——
   P1 的路径同时覆盖落差是否也会被同一反例否决？

解析构造（总计划 §5.1）：U,V ~ iid U(0,1)，L=H=2，预测区间对任意历史恒为 [0.05,0.95]
  （即逐点覆盖恰好 0.90，预测分布完全正确）。
  转变规则：|U-V| > tau，tau = 1 - sqrt(0.05) ⇒ P(转变) = (1-tau)^2 = 0.05
  事后条件覆盖 = (1 - 0.05/sqrt(0.05))^2 = (1-sqrt(0.05))^2 ≈ 0.6028
  由输入/输出独立 ⇒ TypeB 与 TypeC 继承同一数值。

本脚本做三件事：
  (1) 用 Monte Carlo 验证解析值（独立四元组，2e5 与 2e6 两档，报 MC 标准误）
  (2) 同构造下计算"路径同时覆盖"：L=H=2 时路径=两步同时命中，检验其是否
      也能由"把区间换成路径级校准"修复 —— 若能修复，则 P1 的落差与 D3 落差
      属于两类不同性质（可修复的规格选择 vs 不可修复的测量假象）
  (3) 报出修复路径落差所需付出的宽度倍数
用法: python d3_oracle_counterexample.py
"""
import json
import os

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
ROOT = os.path.join(PROJ, 'revision_v1')
A = 0.05
NOM = 0.90
# 转变规则取 |Δ| 的总体 95% 分位 ⇒ P(转变) = (1-tau)^2 = TRANS_P = 0.05
TRANS_P = 0.05
TAU = 1.0 - np.sqrt(TRANS_P)                        # = 1 - sqrt(0.05) = 0.7754
_c = np.sqrt(TRANS_P)                               # 1 - tau
ANALYTIC_POSTHOC = 1.0 - (2 * A * _c - A ** 2) / _c ** 2      # = 0.6028  单端点
# 两步**同时**被覆盖的条件概率（R22-06 补口）：|v2-v1| > tau 且两端点均落在 [A,1-A]
ANALYTIC_POSTHOC_PATH = ((_c - 2 * A) / _c) ** 2             # = 0.305573
assert abs(ANALYTIC_POSTHOC - (1 - A / _c) ** 2) < 1e-12      # 与总计划 §5.1 闭式一致
assert abs(ANALYTIC_POSTHOC_PATH
           - ((np.sqrt(TRANS_P) - 2 * A) / np.sqrt(TRANS_P)) ** 2) < 1e-12


def simulate(n_rep, seed):
    rng = np.random.default_rng(seed)
    # 独立四元组：输入 2 步 (u1,u2)，输出 2 步 (v1,v2)，全部 iid U(0,1)
    u1, u2, v1, v2 = (rng.random(n_rep) for _ in range(4))
    in_tr = np.abs(u2 - u1) > TAU
    out_tr = np.abs(v2 - v1) > TAU
    lab = np.where(in_tr & out_tr, 3, np.where(in_tr, 1, np.where(out_tr, 2, 0)))
    # (1) 逐点区间 [0.05,0.95]：预测分布完全正确
    cov_pt = (v1 >= A) & (v1 <= 1 - A)          # 每个未来端点独立地以 0.90 覆盖
    path_pt = cov_pt & ((v2 >= A) & (v2 <= 1 - A))
    # (2) 路径级区间：把每步半径放大到 r，使两步同时覆盖率达到 0.90
    #     单步 P(hit)=0.90，两步独立 ⇒ 需 (2r)^2 = 0.90 ⇒ r = sqrt(0.90)/2
    r = np.sqrt(NOM) / 2.0
    lo, hi = 0.5 - r, 0.5 + r
    cov_path = (v1 >= lo) & (v1 <= hi) & (v2 >= lo) & (v2 <= hi)
    out = {}
    for g, nm in ((0, 'Normal'), (1, 'TypeA'), (2, 'TypeB'), (3, 'TypeC')):
        m = lab == g
        n = int(m.sum())
        pw = float(cov_pt[m].mean()) if n else float('nan')
        pa = float(path_pt[m].mean()) if n else float('nan')
        se_pw = float(np.sqrt(pw * (1 - pw) / n)) if n else float('nan')
        se_pa = float(np.sqrt(pa * (1 - pa) / n)) if n else float('nan')
        out[nm] = {'n': n,
                   'pointwise_cov': pw,
                   'pathwise_cov_pointwise_intervals': pa,
                   'se_pointwise': se_pw,
                   'se_path': se_pa,
                   'ci95_normal_pointwise': [pw - 1.96 * se_pw, pw + 1.96 * se_pw] if n else None,
                   'ci95_normal_path': [pa - 1.96 * se_pa, pa + 1.96 * se_pa] if n else None}
    return out, float(r * 2 / (1 - 2 * A)), int((lab == 3).sum()), int((lab == 2).sum())


res = {'tau': float(TAU),
       'distribution': 'iid U(0,1) 四元组（输入 2 步、输出 2 步）；**不是**高斯',
       'interval_definition': '逐点区间对任意历史恒为 [A, 1-A] = [0.05, 0.95]，逐步独立覆盖 0.90',
       'transition_rule': f'|v2-v1| > tau = 1 - sqrt({TRANS_P}) ⇒ P(转变) = {TRANS_P}',
       'analytic_posthoc_endpoint_coverage': float(ANALYTIC_POSTHOC),
       'analytic_posthoc_endpoint_formula': '(1 - A/c)^2, c = sqrt(TRANS_P) = 1 - tau',
       'analytic_posthoc_path_coverage': float(ANALYTIC_POSTHOC_PATH),
       'analytic_posthoc_path_formula': '((c - 2A)/c)^2 —— 两步**同时**被覆盖的条件概率',
       'quantities_note': '端点值 ≈0.6028 是"某个未来端点被覆盖"的条件概率；'
                          '两步同时覆盖是另一个量 ≈0.3056。两者不得混称 simultaneous coverage。',
       'runs': {}}
for n_rep, seed in ((200_000, 1), (2_000_000, 2)):
    out, widen, nC, nB = simulate(n_rep, seed)
    res['runs'][f'n={n_rep}'] = {'groups': out, 'width_multiplier_to_fix_pathwise': widen,
                                 'n_TypeC': nC, 'n_TypeB': nB}
    print(f'--- n={n_rep}  tau={TAU:.6f}  解析事后端点覆盖={ANALYTIC_POSTHOC:.6f}')
    for g in ('Normal', 'TypeA', 'TypeB', 'TypeC'):
        d = out[g]
        print(f"    {g:7s} n={d['n']:8d} 逐点={d['pointwise_cov']:.5f}"
              f"(±{1.96*d['se_pointwise']:.5f})  路径(逐点区间)={d['pathwise_cov_pointwise_intervals']:.5f}")
    print(f'    把区间整体放大 {widen:.3f}× 即可让 Normal 的路径同时覆盖回到 0.90')

fn = os.path.join(ROOT, 'results', 'phase0_probes', 'd3_oracle.json')   # 与登记路径一致
json.dump(res, open(fn, 'w', encoding='utf-8'), indent=1)
print('\nsaved ->', fn)
