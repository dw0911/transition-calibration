# -*- coding: utf-8 -*-
"""R1 canonical library synthetic tests (Round1 audit fix acceptance).

Coverage:
  T1-T6  six taxonomy cases (P0-2: includes boundary-spike + prior-window-spike, two old bug killers)
  T7     classify_from_flow vs classify_from_windows cross-path consistency
  T8     conformal_quantile finite-sample correction (P0-6, hand-checked)
  T9     hard-bin q_test discreteness + interp continuity (P0-5)
  T10    valid mask excluded from coverage denominator (P0-3)
  T11    severity fallback shape (P1-3) + Q99 clip (P1-1)
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
sys.path.insert(0, os.path.join(UC, 'SRC'))

import conformal as cf            # noqa: E402
import severity as sev            # noqa: E402
import taxonomy as tx             # noqa: E402

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


# ===========================================================================
print('== T1-T6: six taxonomy cases (tau=10, baseline 100, N=2) ==')
tau = np.array([10.0, 10.0])
IL, OL = 12, 12
g = 24  # target window input start

# T1 fully stable -> Normal
flow = np.full((60, 2), 100.0)
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T1 stable->Normal', lab[0, 0] == tx.NORMAL)

# T2 spike only inside input (ad[g+3]=|x[g+4]-x[g+3]| jumps 500) -> TypeA
flow = np.full((60, 2), 100.0); flow[g + 4, 0] = 600.0
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T2 input-only->TypeA', lab[0, 0] == tx.TYPE_A)

# T3 spike only inside output -> TypeB
flow = np.full((60, 2), 100.0); flow[g + IL + 5, 0] = 600.0
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T3 output-only->TypeB', lab[0, 0] == tx.TYPE_B)

# T4 spike in both input and output -> TypeC
flow = np.full((60, 2), 100.0); flow[g + 2, 0] = 600.0; flow[g + IL + 6, 0] = 600.0
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T4 both->TypeC', lab[0, 0] == tx.TYPE_C)

# T5 boundary-only spike (step: x[g+11]=100 -> x[g+12:]=600 stays; the only jump
#     diff ad[g+11] belongs to neither side) -> Normal
flow = np.full((60, 2), 100.0); flow[g + IL:, 0] = 600.0   # step jump at boundary
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T5 boundary-only->Normal (boundary diff excluded)', lab[0, 0] == tx.NORMAL)

# T6 spike in previous window (ad[g-2]) while current window stable -> Normal
#    (the old 12-step misalignment bug misclassified this)
flow = np.full((60, 2), 100.0); flow[g - 1, 0] = 600.0    # ad[g-2]=|x[g-1]-x[g-2]|=500
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T6 prev-window spike->Normal (P0-2 fix)', lab[0, 0] == tx.NORMAL)

# node 1 unaffected throughout
check('T6b neighbor node Normal', lab[0, 1] == tx.NORMAL)


# ===========================================================================
print('== T7: flow version vs windows version cross-path consistency ==')
rng = np.random.default_rng(0)
flow = 100 + np.cumsum(rng.normal(0, 5, size=(200, 3)), axis=0).clip(min=1)
tau3 = tx.fit_per_node_q95(flow[:100])
g_list = np.arange(100, 150)
lab_flow = tx.classify_from_flow(flow, g_list, tau3, 12, 12)
xw = np.stack([flow[i:i + 12] for i in g_list])       # (n,12,N)
yw = np.stack([flow[i + 12:i + 24] for i in g_list])
lab_win = tx.classify_from_windows(xw, yw, tau3)
check('T7 two-path labels agree', np.array_equal(lab_flow, lab_win))


# ===========================================================================
print('== T8: conformal_quantile finite-sample correction (P0-6) ==')
# n=10, alpha=0.1: k=ceil(11*0.9)=10 -> maximum
s10 = np.arange(1, 11, dtype=float)
check('T8a n=10 -> 10th order statistic (=max)', cf.conformal_quantile(s10, 0.1) == 10.0)
# n=100, alpha=0.1: k=ceil(101*0.9)=91 -> 91st smallest
s100 = np.arange(1, 101, dtype=float)
check('T8b n=100 -> 91st order statistic', cf.conformal_quantile(s100, 0.1) == 91.0)
# old implementation np.quantile(0.9) gives 90.6 (linear interp) != 91; confirm behavior change
check('T8c differs from old np.quantile(0.9)', cf.conformal_quantile(s100, 0.1) != float(np.quantile(s100, 0.9)))


# ===========================================================================
print('== T9: hard vs interpolated (P0-5) ==')
n, H, N = 400, 12, 5
pv = rng.normal(0, 1, (n, H, N)); tv = pv + rng.normal(0, 1, (n, H, N))
pt = rng.normal(0, 1, (200, H, N)); tt = pt + rng.normal(0, 1, (200, H, N))
cv = rng.normal(0, 1, n); ct = rng.normal(0, 1, 200)
labels = rng.integers(0, 4, (200, N)).astype(np.int8)
res_h, meta_h = cf.mondrian_hard(pv, tv, pt, tt, labels, cv, ct, K=5, alpha=0.1)
res_i, meta_i = cf.mondrian_interpolated(pv, tv, pt, tt, labels, cv, ct, K=5, alpha=0.1)
check('T9a hard/interp share the same bin fit', np.allclose(meta_h['q_bins'], meta_i['q_bins']))
check('T9b hard output has q_monotonic', isinstance(res_h['q_monotonic'], bool))
check('T9c coverage finite', 0.5 < res_h['overall_cov'] < 1.0 and 0.5 < res_i['overall_cov'] < 1.0)
check('T9d WCE/GCD/WCE_group present', all(k in res_h for k in ('WCE', 'GCD', 'WCE_group')))


# ===========================================================================
print('== T10: valid mask excluded from coverage denominator (P0-3) ==')
# setup: pred=true (full cover), but half the targets are 0 -> valid coverage
#        should only be computed where true>0
pt2 = np.full((10, 12, 4), 50.0)
tt2 = pt2.copy(); tt2[:, :, :2] = 0.0            # half the nodes target=0 (treated missing)
lab2 = np.zeros((10, 4), dtype=np.int8)          # all Normal
vt2 = tt2 > 0
res2 = cf._eval_groups(pt2, tt2, 20.0, lab2, vt2, 0.9)
check('T10a n_valid=240 (only true>0)', res2['n_valid'] == 240)
check('T10b valid positions all covered -> cov=1', res2['overall_cov'] == 1.0)
# uncond quantile also uses only valid residuals
pv2 = np.full((10, 12, 4), 50.0); tval2 = pv2.copy(); tval2[:, :, :2] = 0.0
res3, q3 = cf.unconditional(pv2, tval2, pt2, tt2, lab2, 0.1, valid_val=tval2 > 0, valid_test=vt2)
check('T10c uncond q=0 (valid residual all 0)', q3 == 0.0)


# ===========================================================================
print('== T11: severity fallback (P1-3) + Q99 clip (P1-1) ==')
# node 0 normal; node 1 all zeros (triggers v.size<10 fallback -> no more IndexError)
ftr = 100 + rng.normal(0, 3, (500, 2)); ftr[:, 1] = 0.0
st = sev.fit_severity_stats(ftr)
check('T11a fallback returns normally with correct shape', st.m.shape == (2,) and st.d.shape == (2,))
# Q99 clip: build extreme s, verify the upper bound of severity_series
s = sev.severity_series(ftr, st)
check('T11b s <= q99 (clip active)', float(s.max()) <= st.q99 + 1e-12)
check('T11c zero diffs set to 0', np.all(s[:, 1] == 0.0))


print(f'\n=== ALL {len(PASS)} TESTS PASSED ===')
