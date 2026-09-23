# -*- coding: utf-8 -*-
"""V=1 完整路径群体的充分统计量（校审任务 2 的统一口径，供核验与重算共用）。

背景（作者校审 2026-09-21，已独立核验）：
  r2a.suff / r2b.suff_from_q 中，路径覆盖与逐点覆盖只统计 12 步全有效的路径（V=1），
  但宽度与 IS 用的是单步有效掩码 vt，即把**不完整路径里的有效预测步**也计入了成本。
  于是同一个"覆盖—成本"点的两条坐标轴对应不同群体。合成检查（宽 2 的完整路径 +
  11 个有效步宽 200 的不完整路径）显示报告均值 96.70 而 V=1 均值应为 2。

本模块提供：
  suff_v1(...)   宽度/IS 也限制在 V=1 群体（其元素即全部 12 步）；
  suff_both(...) 同时返回两套口径，主表用 V1，全有效元素口径保留为辅助列。
不改动 r2a/r2b 的既有函数（旧产物保持可复现），新口径以新字段并行存在。
"""
import numpy as np

GROUPS = {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}


def _core(q, pt, tt):
    pt64, tt64 = pt.astype(np.float64), tt.astype(np.float64)
    half = np.broadcast_to(np.asarray(q, np.float64), pt.shape)
    lo, hi = pt64 - half, pt64 + half
    cov = (tt64 >= lo) & (tt64 <= hi)
    vt = tt64 > 0
    allv = vt.all(axis=1)
    width = 2.0 * half
    pen = (2.0 / 0.10) * np.where(tt64 < lo, lo - tt64, 0.0) + \
          (2.0 / 0.10) * np.where(tt64 > hi, tt64 - hi, 0.0)
    return cov, vt, allv, width, pen


def suff_v1(q, pt, tt, lab):
    """宽度/IS 限制在 V=1 群体（该群体内所有元素均有效）。"""
    cov, vt, allv, width, pen = _core(q, pt, tt)
    m3 = allv[:, None, :]                                   # (n,1,N) → 广播到 (n,H,N)
    s = {'path_hit': (cov.all(axis=1) & allv).sum(axis=1).astype(np.float64),
         'path_elig': allv.sum(axis=1).astype(np.float64),
         'pw_cov': (cov & m3).sum(axis=(1, 2)).astype(np.float64),
         'pw_elig': np.broadcast_to(m3, cov.shape).sum(axis=(1, 2)).astype(np.float64),
         'w_sum': (width * m3).sum(axis=(1, 2)),
         'w_n': np.broadcast_to(m3, cov.shape).sum(axis=(1, 2)).astype(np.float64),
         'is_sum': ((width + pen) * m3).sum(axis=(1, 2))}
    for tid, gn in GROUPS.items():
        sel = (lab == tid) & allv
        s[f'{gn}_hit'] = (cov.all(axis=1) & sel).sum(axis=1).astype(np.float64)
        s[f'{gn}_elig'] = sel.sum(axis=1).astype(np.float64)
    return s


def suff_allvalid(q, pt, tt, lab):
    """旧口径（r2a.suff 等价）：宽度/IS 用单步有效掩码 vt。"""
    cov, vt, allv, width, pen = _core(q, pt, tt)
    s = {'path_hit': (cov.all(axis=1) & allv).sum(axis=1).astype(np.float64),
         'path_elig': allv.sum(axis=1).astype(np.float64),
         'pw_cov': (cov & allv[:, None, :]).sum(axis=(1, 2)).astype(np.float64),
         'pw_elig': np.broadcast_to(allv[:, None, :], cov.shape).sum(axis=(1, 2)).astype(np.float64),
         'w_sum': (width * vt).sum(axis=(1, 2)),
         'w_n': vt.sum(axis=(1, 2)).astype(np.float64),
         'is_sum': ((width + pen) * vt).sum(axis=(1, 2))}
    for tid, gn in GROUPS.items():
        sel = (lab == tid) & allv
        s[f'{gn}_hit'] = (cov.all(axis=1) & sel).sum(axis=1).astype(np.float64)
        s[f'{gn}_elig'] = sel.sum(axis=1).astype(np.float64)
    return s


def suff_both(q, pt, tt, lab):
    return {'V1': suff_v1(q, pt, tt, lab), 'allvalid': suff_allvalid(q, pt, tt, lab)}


def rat(s, num, den):
    d = s[den].sum()
    return float(s[num].sum() / d) if d else float('nan')


def metrics(s):
    return {'path_cov': rat(s, 'path_hit', 'path_elig'),
            'pointwise_same_pop': rat(s, 'pw_cov', 'pw_elig'),
            'mean_width': rat(s, 'w_sum', 'w_n'),
            'IS': rat(s, 'is_sum', 'w_n'),
            'n_paths': float(s['path_elig'].sum()),
            'group_path_cov': {g: rat(s, f'{g}_hit', f'{g}_elig') for g in GROUPS.values()}}
