# -*- coding: utf-8 -*-
"""severity 构造模块 —— UC 课题核心资产（R1 审计修复版，2026-08-17）。

冻结公式（uncalib_v1/PLAN.md + hypotheses/UC-H2.md）::

    severity:      s_{n,t} = |Δx_{n,t} − m_n| / d_n
                   m_n = median(|Δx_n|),  d_n = 1.4826 · MAD_n     （train-only 统计）
    窗口严重度:    r_{i,n}^in  = max_{t ∈ W_in(i)}  s_{n,t}   （past-only，UC-G2 conditioning）
                  r_{i,n}^out = max_{t ∈ W_out(i)} s_{n,t}   （Task E 用，UC 仅诊断对照）

工程约束（冻结，R1 起全部实施）:
  - 零值 mask：差分两端任一时刻 flow=0 → 该差分不参与统计、序列上置 0
  - 99 分位 clip 前置：s ← min(s, Q99(s_train))（P1-1 修复：代码与协议对齐；
    r = max over window ≤ Q99 自然继承，等价于协议中的 r ← min(r, Q99(r_train))）
  - τ / clip 分位均取 train 统计；val/test 只应用不重估（无泄漏）

R1 修复（Round1 审计）:
  - P1-3: fit_severity_stats fallback 索引改为 ad[ok]（2D bool 语义正确）
  - P1-1: Q99 clip 实际实施（此前仅文档声明）
  - P1-2 备注: s 为双侧偏离 |Δx|−m（change-magnitude anomaly severity，冻结定义保留；
    单侧敏感性对照见 ucg1c 系）

部署边界（UC-G2 冻结）：区间宽度只依赖历史可见 r^in，推理时不读 future TypeC/severity。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeverityStats:
    m: np.ndarray      # (N,) per-node median of |Δx|（train, 零值 mask）
    d: np.ndarray      # (N,) per-node 1.4826*MAD of |Δx|（train, 零值 mask）
    q99: float = 0.0   # train 段 valid s 的 99 分位（clip 上界，P1-1）

    def asdict(self):
        return {'m': self.m.tolist(), 'd': self.d.tolist(), 'q99': self.q99}


def _absdiff_masked(flow):
    """|Δx| 与有效位 mask。flow (T,N) → ad (T-1,N), ok (T-1,N) bool。"""
    x = np.asarray(flow, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=0))
    ok = (x[1:] > 0) & (x[:-1] > 0)
    return ad, ok


def fit_severity_stats(flow_train):
    """从 train 段拟合 per-node robust 统计（零值 mask）+ 全局 Q99 clip 上界。"""
    ad, ok = _absdiff_masked(flow_train)
    _, N = ad.shape
    m = np.zeros(N); d = np.ones(N)
    for n in range(N):
        v = ad[ok[:, n], n]                       # 该节点 valid diffs
        if v.size < 10:
            v = ad[ok]                            # P1-3 修复：全局 valid fallback（2D bool 索引）
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        m[n] = med
        d[n] = 1.4826 * mad if mad > 0 else max(np.std(v), 1e-6)
    stats = SeverityStats(m=m, d=d)
    # P1-1: train 段 valid s 的 Q99（clip 上界；零值位置 s=0 不参与分位）
    s_train = np.where(ok, np.abs(ad - m) / d, 0.0)
    valid_s = s_train[ok]
    stats.q99 = float(np.quantile(valid_s, 0.99)) if valid_s.size > 0 else float('inf')
    return stats


def severity_series(flow, stats):
    """全序列 severity s (T-1, N)。零值差分置 0；P1-1: clip 到 train Q99。"""
    ad, ok = _absdiff_masked(flow)
    s_raw = np.abs(ad - stats.m) / stats.d
    s = np.where(ok, np.minimum(s_raw, stats.q99), 0.0)   # 零值置 0 + Q99 clip
    return s


def window_in_severity(s, g_indices, input_len=12):
    """r^in (n_win, N)：W_in 差分索引 [g, g+input_len-1)。g = input 起点。"""
    g = np.asarray(g_indices)
    ar = np.arange(input_len - 1)
    blk = s[g[:, None] + ar] if len(g) else s[:0]    # (n_win, IN-1, N)
    return blk.max(axis=1)


def window_out_severity(s, p_indices, output_len=12):
    """r^out (n_win, N)：W_out 差分索引 [p, p+output_len-1)。p = output 起点。"""
    p = np.asarray(p_indices)
    ar = np.arange(output_len - 1)
    blk = s[p[:, None] + ar] if len(p) else s[:0]
    return blk.max(axis=1)


def window_in_severity_from_windows(x_win, stats, input_len=12):
    """窗口数组版 r^in（legacy npz 路径，无需序列锚点）。

    x_win: (n, input_len, N) raw flow。与 window_in_severity 在相同窗口上严格等价
    （r^in 只依赖窗口内差分）。P0-4 修复：STID 等窗口数据路径复用同一 severity 定义。
    """
    x = np.asarray(x_win, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=1))                     # (n, IL-1, N)
    ok = (x[:, 1:] > 0) & (x[:, :-1] > 0)
    s = np.where(ok, np.minimum(np.abs(ad - stats.m) / stats.d, stats.q99), 0.0)
    return s.max(axis=1)
