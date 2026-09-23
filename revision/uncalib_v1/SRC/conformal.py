# -*- coding: utf-8 -*-
"""Conformal calibration canonical library —— Round1 审计修复版（R1，2026-08-17）。

修复对应审计项：
  P0-5: mondrian_hard（主方法，标准 Mondrian 组内保证）与 mondrian_interpolated
        （smooth variant，工程增强，无标准有限样本保证）显式拆分
  P0-6: conformal_quantile 实现预注册有限样本修正 k=⌈(n+1)(1-α)⌉（method="higher"）
  P0-3: valid mask（true>0）贯穿 residual quantile / coverage / width / WCE / GCD
  P1-4: q 单调性检查（记录 q_monotonic 元数据，不强制篡改）

口径约定（全库统一）：
  - conformal score 单位：node-horizon (window, horizon, node)
  - coverage / WCE / GCD 只在 valid_target（默认 true>0）上计算
  - residual quantile 只用 valid residual
  - 报告口径为 empirical node-horizon coverage（时序/空间相关性存在，不作 guarantee 声明）
"""
import numpy as np

# ---------------------------------------------------------------------------
# P0-6: 预注册有限样本修正分位
# ---------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    """split conformal 分位数：scores 的第 k=⌈(n+1)(1-α)⌉ 个次序统计量（升序 1-based）。

    k>n（小样本下 α 不足 1/n）时收缩到 n（取最大值，有限样本正确行为）。
    直接用排序取 rank，避免 np.quantile 线性插值位置的 off-by-one
    （h=(n-1)·level 经 'higher' 会指向 rank+1，T8b 测试捕获）。
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        raise ValueError('conformal_quantile: empty scores')
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    return float(np.sort(scores)[k - 1])


# ---------------------------------------------------------------------------
# 评价（P0-3: valid mask 贯穿）
# ---------------------------------------------------------------------------

def _eval_groups(pred, true, q, labels, valid_target=None, nominal=0.9):
    """per-group coverage/width/gap + WCE/GCD。

    pred/true: (n,H,N)；q: 标量或 (n,H,N)；labels: (n,N) int8；
    valid_target: (n,H,N) bool，None 视为全有效（调用方负责 true>0 语义）。
    """
    covered = (true >= pred - q) & (true <= pred + q)
    width = 2 * (q if np.isscalar(q) else q)
    if valid_target is None:
        valid_target = np.ones(covered.shape, dtype=bool)
    else:
        valid_target = np.asarray(valid_target, dtype=bool)
    n_valid_all = int(valid_target.sum())
    out = {'n_valid': n_valid_all, 'n_total': int(valid_target.size)}
    if n_valid_all > 0:
        out['overall_cov'] = float((covered & valid_target).sum() / n_valid_all)
        out['mean_width'] = float((width * valid_target).sum() / n_valid_all)
    else:
        out['overall_cov'] = float('nan')
        out['mean_width'] = float('nan')
    cn = None
    for tid, gn in {0: 'Normal', 1: 'TypeA', 2: 'TypeB', 3: 'TypeC'}.items():
        sel = (labels == tid)[:pred.shape[0]]                      # (n,N)
        gmask = np.broadcast_to(np.expand_dims(sel, 1), covered.shape) & valid_target
        n = int(gmask.sum())
        if n == 0:
            out[gn] = {'cov': float('nan'), 'width': float('nan'), 'n': 0}
            continue
        cov = float((covered & gmask).sum() / n)
        gw = float(width) if np.isscalar(q) else float((width * gmask).sum() / n)
        out[gn] = {'cov': cov, 'gap': cov - nominal, 'width': gw, 'n': n}
        if tid == 0:
            cn = cov
    if cn is not None:
        for gn in ['TypeA', 'TypeB', 'TypeC']:
            if gn in out and not np.isnan(out[gn]['cov']):
                out[gn]['cond_gap'] = out[gn]['cov'] - cn
    covs = [out[g]['cov'] for g in ['Normal', 'TypeA', 'TypeB', 'TypeC']
            if g in out and not np.isnan(out[g]['cov'])]
    if covs:
        out['WCE'] = float(max(abs(c - nominal) for c in covs))
        out['GCD'] = float(max(covs) - min(covs))
        # WCE 责任组（P0-C 审计项）
        gaps = {g: abs(out[g]['cov'] - nominal) for g in ['Normal', 'TypeA', 'TypeB', 'TypeC']
                if g in out and not np.isnan(out[g]['cov'])}
        out['WCE_group'] = max(gaps, key=gaps.get)
    return out


# ---------------------------------------------------------------------------
# unconditional
# ---------------------------------------------------------------------------

def unconditional(pred_val, true_val, pred_test, true_test, labels, alpha=0.1,
                  valid_val=None, valid_test=None):
    """全局 conformal（valid residual 估 q）。返回 (res, q)。"""
    nominal = 1.0 - alpha
    R = np.abs(pred_val - true_val)
    if valid_val is not None:
        R = R[np.asarray(valid_val, dtype=bool)]
    q = conformal_quantile(R.ravel(), alpha)
    return _eval_groups(pred_test, true_test, q, labels, valid_test, nominal), q


# ---------------------------------------------------------------------------
# Mondrian（P0-5: hard 主方法 / interpolated 变体）
# ---------------------------------------------------------------------------

def _broadcast_cond(cond, shape):
    """cond (n,) 或 (n,N) → (n,H,N)。"""
    cond = np.asarray(cond)
    if cond.ndim == 1:
        return np.broadcast_to(cond[:, None, None], shape)
    return np.broadcast_to(cond[:, None, :], shape)


def _bin_fit(pred_val, true_val, cond_val, K, alpha, valid_val=None):
    """桶拟合：edges / anchors / q_bins（每桶 conformal_quantile）/ 单调性 flag。"""
    R = np.abs(pred_val - true_val)                                # (n,H,N)
    cv = _broadcast_cond(cond_val, R.shape)
    if valid_val is not None:
        vm = np.asarray(valid_val, dtype=bool)
    else:
        vm = np.ones(R.shape, dtype=bool)
    Rf, cvf = R[vm], cv[vm]
    if Rf.size == 0:
        raise ValueError('_bin_fit: no valid calibration elements')
    edges = np.quantile(cvf, np.linspace(0, 1, K + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    bidx = np.digitize(cvf, edges[1:-1])                           # 0..K-1
    gq = conformal_quantile(Rf, alpha)                             # 小桶 fallback
    q_bins, anchors = np.zeros(K), np.zeros(K)
    for k in range(K):
        m = bidx == k
        q_bins[k] = conformal_quantile(Rf[m], alpha) if int(m.sum()) > 10 else gq
        anchors[k] = float(np.median(cvf[m])) if int(m.sum()) > 0 else float(edges[k])
    order = np.argsort(anchors)
    anchors, q_bins = anchors[order], q_bins[order]
    mono = bool(np.all(np.diff(q_bins) >= -1e-12))                 # P1-4: 检查不篡改
    return edges, anchors, q_bins, mono


def mondrian_hard(pred_val, true_val, pred_test, true_test, labels, cond_val, cond_test,
                  K=5, alpha=0.1, valid_val=None, valid_test=None):
    """主方法：hard-bin Mondrian conformal。

    每个桶 k 用桶内 conformal_quantile(n_k)；test 点按其条件变量所属桶取 q_k
    （不插值、不外推——超出 edges 的 clamp 到首末桶，属桶定义内）。
    返回 (res, meta)，meta 含 edges/anchors/q_bins/q_monotonic/每桶 n。
    """
    nominal = 1.0 - alpha
    edges, anchors, q_bins, mono = _bin_fit(pred_val, true_val, cond_val, K, alpha, valid_val)
    ct = _broadcast_cond(cond_test, pred_test.shape)
    b = np.clip(np.digitize(ct, edges[1:-1]), 0, K - 1)
    q_test = q_bins[b]
    res = _eval_groups(pred_test, true_test, q_test, labels, valid_test, nominal)
    res['q_monotonic'] = mono
    meta = {'edges': edges.tolist(), 'anchors': anchors.tolist(),
            'q_bins': q_bins.tolist(), 'q_monotonic': mono, 'K': K}
    return res, meta


def mondrian_interpolated(pred_val, true_val, pred_test, true_test, labels, cond_val, cond_test,
                          K=5, alpha=0.1, valid_val=None, valid_test=None):
    """smooth variant：相邻桶 anchors 上线性插值（np.interp，端点值=不外推）。

    注意：插值点的 q 不再等于其所属桶的正式 conformal quantile，
    标准 Mondrian 有限样本保证不自动保留——仅作工程增强报告。
    """
    nominal = 1.0 - alpha
    edges, anchors, q_bins, mono = _bin_fit(pred_val, true_val, cond_val, K, alpha, valid_val)
    ct = _broadcast_cond(cond_test, pred_test.shape)
    q_test = np.interp(ct, anchors, q_bins)
    res = _eval_groups(pred_test, true_test, q_test, labels, valid_test, nominal)
    res['q_monotonic'] = mono
    meta = {'edges': edges.tolist(), 'anchors': anchors.tolist(),
            'q_bins': q_bins.tolist(), 'q_monotonic': mono, 'K': K}
    return res, meta


# ---------------------------------------------------------------------------
# 三色（项目管理 Gate，非论文指标——审计 P0 结论保留使用）
# ---------------------------------------------------------------------------

def three_color(res_method, res_uncond):
    cov_tc = res_method['TypeC']['cov']
    cov_uncond_tc = res_uncond['TypeC']['cov']
    gain = cov_tc - cov_uncond_tc
    width_ratio = res_method['mean_width'] / res_uncond['mean_width']
    if (cov_tc >= 0.90 or gain >= 0.10) and width_ratio <= 1.5:
        color = 'Green'
    elif 0.05 <= gain < 0.10 and width_ratio <= 1.3:
        color = 'Yellow'
    else:
        color = 'Red'
    return {'color': color, 'gain': gain, 'width_ratio': width_ratio,
            'cov_TypeC': cov_tc, 'cov_TypeC_uncond': cov_uncond_tc}
