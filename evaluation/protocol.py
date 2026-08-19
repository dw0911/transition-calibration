# -*- coding: utf-8 -*-
"""Evaluation protocol canonical facade -- unified evaluation entry point (R1 canonical system).

All scripts must call this module (+ severity/conformal/taxonomy); implementing evaluation
logic locally is forbidden. Corresponding audit items:
  R1 taxonomy alignment (unified window-difference boundary definition)
  R1 valid mask consistency across the whole pipeline
  R0 scale consistency (inference protocol: per-node Z-score done once before inference)

Frozen conventions (consistent with PLAN/hypotheses):
  - window g (input start): X=x[g:g+L], Y=x[g+L:g+L+H]
  - input diffs  = [g, g+L-1); output diffs = [g+L, g+L+H-1); boundary diff excluded
  - valid_diff(t) = I(x_t>0) * I(x_{t+1}>0)
  - valid_target(i,n,h) = I(y>0) AND isfinite(y)
  - alpha = 0.10 frozen throughout
"""
import numpy as np

ALPHA = 0.10
INPUT_LEN = 12
OUTPUT_LEN = 12


def compute_valid_target(true):
    """(n,H,N) -> valid_target mask (true>0 AND finite)."""
    return (np.isfinite(true)) & (true > 0)


def compute_valid_diff(flow):
    """(T,N) -> valid_diff (T-1,N) (both endpoints >0)."""
    x = np.asarray(flow, dtype=np.float64)
    return (x[1:] > 0) & (x[:-1] > 0)


def fit_per_node_q95(flow_train):
    """per-node Q95(|dx|), using only valid diffs. Delegates to taxonomy."""
    from calibration.taxonomy import fit_per_node_q95 as _fit
    return _fit(flow_train)


def classify_windows(x_win, y_win, tau):
    """Window-version taxonomy (legacy npz / STID path). Delegates to taxonomy."""
    from calibration.taxonomy import classify_from_windows
    return classify_from_windows(x_win, y_win, tau)


def classify_flow(flow, g_indices, tau):
    """Full-series taxonomy (BasicTS / STAEformer path). Delegates to taxonomy."""
    from calibration.taxonomy import classify_from_flow
    return classify_from_flow(flow, g_indices, tau, INPUT_LEN, OUTPUT_LEN)


def evaluate_groups(pred, true, q, labels, valid_target=None):
    """per-group coverage/width/WCE/GCD. Delegates to conformal."""
    from calibration.mondrian import _eval_groups
    return _eval_groups(pred, true, q, labels, valid_target, 1.0 - ALPHA)


def infer_protocol(mean, std, x_raw):
    """Inference normalization protocol (audit item R0): per-node Z-score done once before inference.

    Returns normalized x (float32 numpy). The model consumes norm; outputs are inverse-transformed
    once via out*std+mean.
    """
    x = x_raw.copy()
    x[..., 0] = (x[..., 0] - mean) / std
    return x
