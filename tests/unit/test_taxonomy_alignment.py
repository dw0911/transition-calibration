# -*- coding: utf-8 -*-
"""R1: taxonomy alignment six synthetic cases (T1-T6, Round1 audit P0-2 acceptance).

Window g (input start): input diffs=[g, g+11), output diffs=[g+12, g+23), boundary diff excluded.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[2]
UC = REPRO_ROOT
import calibration.taxonomy as tx  # noqa

PASS = []
def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')

IL, OL = 12, 12
tau = np.array([10.0, 10.0])
g = 24

flow = np.full((60, 2), 100.0)
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T1 stable->Normal', lab[0, 0] == tx.NORMAL)

flow = np.full((60, 2), 100.0); flow[g + 4, 0] = 600.0
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T2 input-only->TypeA', lab[0, 0] == tx.TYPE_A)

flow = np.full((60, 2), 100.0); flow[g + IL + 5, 0] = 600.0
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T3 output-only->TypeB', lab[0, 0] == tx.TYPE_B)

flow = np.full((60, 2), 100.0); flow[g + 2, 0] = 600.0; flow[g + IL + 6, 0] = 600.0
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T4 both->TypeC', lab[0, 0] == tx.TYPE_C)

flow = np.full((60, 2), 100.0); flow[g + IL:, 0] = 600.0  # step jump only at boundary
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T5 boundary-only->Normal', lab[0, 0] == tx.NORMAL)

flow = np.full((60, 2), 100.0); flow[g - 1, 0] = 600.0  # prior-window spike
lab = tx.classify_from_flow(flow, [g], tau, IL, OL)
check('T6 prev-window->Normal (P0-2 fix)', lab[0, 0] == tx.NORMAL)
check('T6b neighbor node Normal', lab[0, 1] == tx.NORMAL)

if __name__ == '__main__':
    print(f'[test_taxonomy_alignment] {len(PASS)}/6 PASS')
