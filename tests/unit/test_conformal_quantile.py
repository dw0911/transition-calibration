# -*- coding: utf-8 -*-
"""R1: finite-sample conformal quantile (P0-6 acceptance).

k=ceil((n+1)(1-alpha)) corresponds to the order statistic; not NumPy's default
linear interpolation.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
sys.path.insert(0, os.path.join(UC, 'SRC'))
import conformal as cf  # noqa

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


# n=10, alpha=0.1: k=ceil(11*0.9)=10 -> 10th order statistic
s10 = np.arange(1, 11, dtype=float)
check('n=10 -> max (=10th order stat)', cf.conformal_quantile(s10, 0.1) == 10.0)
# n=100, alpha=0.1: k=ceil(101*0.9)=91 -> 91st smallest
s100 = np.arange(1, 101, dtype=float)
check('n=100 -> 91th order stat (=91)', cf.conformal_quantile(s100, 0.1) == 91.0)
# differs from old np.quantile(0.9) linear interpolation
check('differs from np.quantile(0.9) interp', cf.conformal_quantile(s100, 0.1) != float(np.quantile(s100, 0.9)))
# small-sample k>n shrinks to max
s5 = np.arange(1, 6, dtype=float)
check('n=5 small-sample shrink to max', cf.conformal_quantile(s5, 0.1) == 5.0)

if __name__ == '__main__':
    print(f'[test_conformal_quantile] ALL {len(PASS)} PASS')
