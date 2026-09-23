# -*- coding: utf-8 -*-
"""E5b：λ 收缩是否可部署 —— 只用 Selection 选 λ*，再看 test 上是否仍占优。

E5 的 λ 扫描出现一个未预期的结果：在全部 4 个 config、3 个方法族上，
λ=0.75（把条件尺度向常数收缩 25%）在 **覆盖与宽度两个目标上同时优于** λ=1.0（完全条件化）。
若成立，这不是观察而是方法：**收缩正则化的条件 conformal**。

但 λ=0.75 是我看了 test 结果后注意到的，属事后挑选。本脚本把它变成可部署规则：
  选择规则（只用 Selection 段，禁止 test）：
      λ* = argmin_λ  mean_width_sel(λ)   s.t.  path_cov_sel(λ) ≥ 0.90 − tol_sel
  其中 q_λ 仍在 Calibration 上按名义 α 拟合（与主协议一致），Selection 只用于挑 λ。
  tol_sel = 0.005（预登记，不事后调整）。

对照：λ=1（完全条件化，即 E5/r2b 的发表口径方法）。
输出：每 config × 每方法族的 λ*、Selection 与 Test 上的覆盖/宽度、以及 λ* 相对 λ=1 的
      test 配对差值（window 级平稳块 bootstrap，2000 次）。
判定（预登记）：
  PASS  —— 若 λ* 由 Selection 选出后，在 test 上仍满足 (覆盖不低于 λ=1 − 0.005) 且 (宽度更小)，
           且该结论在 ≥3/4 config 上成立
  PARTIAL —— 方向成立但不足 3/4
  FAIL  —— λ* 在 test 上不占优（说明 E5 的 λ=0.75 优势是事后挑选的产物）
用法: python scripts/e5b_shrinkage.py
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
import e5_drift_penalty as e5                         # noqa: E402

ALPHA, NOM = 0.10, 0.90
TOL_SEL = 0.005           # 预登记
LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
N_BOOT, P_RESTART = 2000, 1.0 / 288.0
SEEDS = [('PEMS04', 42), ('PEMS04', 123), ('PEMS04', 456), ('PEMS03', 42)]


def boot_index(n, rng):
    idx = np.empty(n, dtype=np.int64); f = 0
    while f < n:
        st = int(rng.integers(n)); L = int(rng.geometric(P_RESTART))
        b = (st + np.arange(L)) % n; t = min(L, n - f)
        idx[f:f + t] = b[:t]; f += t
    return idx


def half_for(f_cal, f_sel, f_test, Sc, vc, u, lam):
    """给定 λ，在 calib 上拟合 q，返回 sel/test 两段半宽。"""
    mref = float(f_cal[vc].mean())
    fcl = (1 - lam) + lam * (f_cal / mref)
    q = cf.conformal_quantile((Sc / fcl)[vc], ALPHA)
    ub = u[None, :, :] if np.ndim(u) == 2 else 1.0
    return q, q * ((1 - lam) + lam * (f_sel / mref))[:, None, :] * ub, \
        q * ((1 - lam) + lam * (f_test / mref))[:, None, :] * ub


def eval_seg(half, p, t):
    half = np.broadcast_to(np.asarray(half, np.float64), p.shape)
    p64, t64 = p.astype(np.float64), t.astype(np.float64)
    cov = (t64 >= p64 - half) & (t64 <= p64 + half)
    v = t64 > 0
    allv = v.all(axis=1)
    return {'path_cov': float((cov.all(axis=1) & allv).sum() / max(allv.sum(), 1)),
            'mean_width': float((2 * half)[v].mean()),
            'hit': (cov.all(axis=1) & allv).sum(axis=1).astype(np.float64),
            'elig': allv.sum(axis=1).astype(np.float64),
            'wsum': (2 * half * v).sum(axis=(1, 2)).astype(np.float64),
            'wn': v.sum(axis=(1, 2)).astype(np.float64)}


def run(ds, seed, out):
    t0 = time.time()
    dsel, dcal, dtest = (e5.load(ds, seed, x) for x in ('sel', 'cal', 'test'))
    g = r2b.g_ranges(ds)
    a, _ = r2b.fit_scale(dsel)
    n = dtest['pred'].shape[0]; N = a.shape[1]
    Sc_c, vc = e5.path_scores(dcal, 1.0)
    Sca_c, _ = e5.path_scores(dcal, a)
    fams = {}
    # severity Mondrian（u=1）
    fc, _ = e5.mondrian_f(dcal['r_in_max'].astype(np.float64),
                          dtest['r_in_max'].astype(np.float64), Sc_c, vc, 5)
    fs = e5.mondrian_f(dcal['r_in_max'].astype(np.float64),
                       dsel['r_in_max'].astype(np.float64), Sc_c, vc, 5)[1]
    ft = e5.mondrian_f(dcal['r_in_max'].astype(np.float64),
                       dtest['r_in_max'].astype(np.float64), Sc_c, vc, 5)[1]
    fams['path_sev_K5'] = (fc, fs, ft, Sc_c, 1.0)
    # g0 / g1（u=a）
    Xs = {0: r2b.features(dsel, g['sel'], a, False), 1: r2b.features(dsel, g['sel'], a, True)}
    Xc = {0: r2b.features(dcal, g['cal'], a, False), 1: r2b.features(dcal, g['cal'], a, True)}
    Xt = {0: r2b.features(dtest, g['test'], a, False), 1: r2b.features(dtest, g['test'], a, True)}
    ys, vs = r2b.path_risk(dsel, a); vsf = vs.ravel()
    for j, nm in ((0, 'g0_base'), (1, 'g1_base_plus_severity')):
        mdl = HistGradientBoostingRegressor(random_state=seed, **r2b.LEARNER).fit(Xs[j][vsf], ys)
        gc_ = np.maximum(mdl.predict(Xc[j]), 0.0).reshape(dcal['pred'].shape[0], N)
        gs_ = np.maximum(mdl.predict(Xs[j]), 0.0).reshape(dsel['pred'].shape[0], N)
        gt_ = np.maximum(mdl.predict(Xt[j]), 0.0).reshape(n, N)
        fl = r2b.G_FLOOR_REL * float(np.median(gc_[vc]))
        med = float(np.median(gc_[vc]))
        fams[nm] = (np.maximum(gc_, fl) / med, np.maximum(gs_, fl) / med,
                    np.maximum(gt_, fl) / med, Sca_c, a)
        del mdl
    entry = {}
    rng = np.random.default_rng(31)
    for fam, (fcx, fsx, ftx, Sc, u) in fams.items():
        rows = []
        for lam in LAM_GRID:
            q, hs, ht = half_for(fcx, fsx, ftx, Sc, vc, u, lam)
            es = eval_seg(hs, dsel['pred'], dsel['true'])
            et = eval_seg(ht, dtest['pred'], dtest['true'])
            rows.append({'lambda': lam, 'q': float(q),
                         'sel_path_cov': es['path_cov'], 'sel_width': es['mean_width'],
                         'test_path_cov': et['path_cov'], 'test_width': et['mean_width'],
                         '_es': es, '_et': et})
        elig = [r for r in rows if r['sel_path_cov'] >= NOM - TOL_SEL]
        best = min(elig, key=lambda r: r['sel_width']) if elig else min(rows, key=lambda r: r['sel_width'])
        full = rows[-1]
        dc, wr = [], []
        for _ in range(N_BOOT):
            bi = boot_index(n, rng)
            eb, fb = best['_et'], full['_et']
            cb = eb['hit'][bi].sum() / eb['elig'][bi].sum()
            cf_ = fb['hit'][bi].sum() / fb['elig'][bi].sum()
            wb = eb['wsum'][bi].sum() / eb['wn'][bi].sum()
            wf = fb['wsum'][bi].sum() / fb['wn'][bi].sum()
            dc.append(cb - cf_); wr.append(wb / wf)
        dc, wr = np.array(dc), np.array(wr)
        entry[fam] = {
            'lambda_star': best['lambda'], 'lambda_star_rule':
                f'argmin sel_width s.t. sel_path_cov >= {NOM - TOL_SEL}',
            'n_lambda_eligible': len(elig),
            'selection': {'path_cov': best['sel_path_cov'], 'width': best['sel_width']},
            'test_at_lambda_star': {'path_cov': best['test_path_cov'], 'width': best['test_width']},
            'test_at_lambda_1': {'path_cov': full['test_path_cov'], 'width': full['test_width']},
            'delta_cov_star_minus_full': best['test_path_cov'] - full['test_path_cov'],
            'width_ratio_star_over_full': best['test_width'] / full['test_width'],
            'delta_cov_CI95': [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
            'width_ratio_CI95': [float(np.percentile(wr, 2.5)), float(np.percentile(wr, 97.5))],
            'grid': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]}
        for r in rows:
            r.pop('_es', None); r.pop('_et', None)
    out[f'{ds}_s{seed}'] = entry
    print(f'\n===== {ds} s{seed} ({time.time()-t0:.0f}s)', flush=True)
    for fam, e in entry.items():
        print(f"  {fam:22s} λ*={e['lambda_star']:.2f} (合格 λ 数 {e['n_lambda_eligible']}/{len(LAM_GRID)}) "
              f"| test: cov {e['test_at_lambda_1']['path_cov']:.4f}(λ=1) → "
              f"{e['test_at_lambda_star']['path_cov']:.4f}(λ*)  Δ={e['delta_cov_star_minus_full']:+.4f} "
              f"CI[{e['delta_cov_CI95'][0]:+.4f},{e['delta_cov_CI95'][1]:+.4f}] | "
              f"width ratio={e['width_ratio_star_over_full']:.4f} "
              f"CI[{e['width_ratio_CI95'][0]:.4f},{e['width_ratio_CI95'][1]:.4f}]", flush=True)
    return entry


def main():
    out = {'rule': 'lambda* = argmin sel_width s.t. sel_path_cov >= 0.895 (TOL_SEL=0.005, 预登记); '
                   'q_lambda 仍在 Calibration 上按名义 alpha 拟合；Selection 只用于挑 lambda',
           'verdict_criteria': {'PASS': 'λ* 在 test 上满足 覆盖不低于 λ=1 − 0.005 且 宽度更小，'
                                        '且 ≥3/4 config 成立',
                                'PARTIAL': '方向成立但 <3/4 config',
                                'FAIL': 'λ* 在 test 上不占优 → E5 的 λ=0.75 优势属事后挑选'},
           'configs': {}}
    for ds, sd in SEEDS:
        run(ds, sd, out['configs'])
    wins, dirs = 0, 0
    for c, e in out['configs'].items():
        for fam, v in e.items():
            ok = (v['delta_cov_star_minus_full'] >= -0.005) and (v['width_ratio_star_over_full'] < 1.0)
            wins += int(ok)
            dirs += int(v['width_ratio_star_over_full'] < 1.0)
    tot = sum(len(e) for e in out['configs'].values())
    out['verdict'] = {'n_family_config': tot, 'n_pass': wins, 'n_width_smaller': dirs,
                      'result': 'PASS' if wins >= int(0.75 * tot) else ('PARTIAL' if wins > 0 else 'FAIL')}
    fn = os.path.join(ROOT, 'results', 'phase1_R1', 'e5b_shrinkage.json')
    json.dump(out, open(fn, 'w', encoding='utf-8'), indent=1)
    print(f"\n===== 判定：{out['verdict']['result']}  "
          f"(通过 {wins}/{tot} 个 family×config；宽度更小 {dirs}/{tot})")
    print('saved ->', fn)


if __name__ == '__main__':
    main()
