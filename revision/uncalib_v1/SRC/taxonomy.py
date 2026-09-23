# -*- coding: utf-8 -*-
"""Transition taxonomy canonical library —— R1 审计修复版（2026-08-17）。

Round1 审计修复：
  P0-2: 修正旧四段实现的 12 步时间错位。窗口 g（input 起点）的标签对齐：
        input diffs  = gdiff[g   : g+IL-1]      （input 窗口内 11 个差分）
        output diffs = gdiff[g+IL: g+IL+OL-1]   （output 窗口内 11 个差分）
        边界差分（g+IL-1 → g+IL，即 input 末点→output 首点）不计入，
        与 legacy np.diff(input)/np.diff(output) 语义一致。
  P0-3: 零值 mask 统一——tau 拟合与标签判定均只用 valid diffs（两端 flow>0），
        0↔x 传感器关断不再被误判为 transition。

标签语义（冻结）：
  0=Normal（in/out 均无突变） 1=TypeA（仅 in） 2=TypeB（仅 out） 3=TypeC（in+out）
"""
import numpy as np

NORMAL, TYPE_A, TYPE_B, TYPE_C = 0, 1, 2, 3


def _diffs_masked(flow):
    """|Δx| 与 valid mask。flow (T,N) → ad (T-1,N), ok (T-1,N)。"""
    x = np.asarray(flow, dtype=np.float64)
    ad = np.abs(np.diff(x, axis=0))
    ok = (x[1:] > 0) & (x[:-1] > 0)
    return ad, ok


def fit_per_node_q95(flow_train):
    """per-node Q95(|Δx|)，只用 valid diffs（P0-3）。返回 tau (N,)。"""
    ad, ok = _diffs_masked(flow_train)
    _, N = ad.shape
    tau = np.zeros(N)
    for n in range(N):
        v = ad[ok[:, n], n]
        if v.size < 10:
            v = ad[ok]
        tau[n] = np.quantile(v, 0.95)
    return tau


def classify_from_flow(flow, g_indices, tau, input_len=12, output_len=12):
    """raw 全序列版（四段 STAEformer/BasicTS 路径）。

    flow (T,N)；g_indices = 各窗口 input 起点 g。返回 labels (n_win, N) int8。
    input diffs [g, g+IL-1)；output diffs [g+IL, g+IL+OL-1)；边界差分不计。
    """
    ad, ok = _diffs_masked(flow)
    g = np.asarray(g_indices)
    n_win = len(g)
    labels = np.zeros((n_win, flow.shape[1]), dtype=np.int8)
    ar_in = np.arange(input_len - 1)
    ar_out = np.arange(output_len - 1)
    for i, gi in enumerate(g):
        # (IL-1,N) / (OL-1,N)，超范围窗口由调用方保证合法
        hi = ((ad[gi + ar_in] > tau) & ok[gi + ar_in]).any(axis=0)
        ho = ((ad[gi + input_len + ar_out] > tau) & ok[gi + input_len + ar_out]).any(axis=0)
        labels[i, hi & ~ho] = TYPE_A
        labels[i, ~hi & ho] = TYPE_B
        labels[i, hi & ho] = TYPE_C
    return labels


def classify_from_windows(x_win, y_win, tau):
    """legacy npz 窗口版（UC-G1 三段/STID 路径，语义本来就正确，此处补 zero mask）。

    x_win (n,IL,N)，y_win (n,OL,N)。窗口内 np.diff，边界差分不存在。
    """
    fx = np.asarray(x_win, dtype=np.float64)
    fy = np.asarray(y_win, dtype=np.float64)
    gin = np.abs(np.diff(fx, axis=1)) > tau[None, None, :]
    vin = (fx[:, 1:] > 0) & (fx[:, :-1] > 0)
    gin &= vin
    gout = np.abs(np.diff(fy, axis=1)) > tau[None, None, :]
    vout = (fy[:, 1:] > 0) & (fy[:, :-1] > 0)
    gout &= vout
    hi = gin.any(axis=1); ho = gout.any(axis=1)
    labels = np.zeros((fx.shape[0], fx.shape[2]), dtype=np.int8)
    labels[hi & ~ho] = TYPE_A
    labels[~hi & ho] = TYPE_B
    labels[hi & ho] = TYPE_C
    return labels
