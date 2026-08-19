# Beyond Marginal Coverage: Transition-Aware Uncertainty Calibration for Reliable Spatio-Temporal Forecasting

This repository provides the official implementation of the uncertainty-calibration framework
described in the paper. The core idea is that *marginal* conformal coverage is not enough for
reliable spatio-temporal forecasting: when the traffic state undergoes an abrupt transition
(e.g., sensor dropout or sudden congestion onset), the point forecast and its residual
distribution both shift. The framework detects those transitions from **past-observable**
information only and calibrates the prediction intervals accordingly.

## What is included

- **Transition failure taxonomy** (`calibration/taxonomy.py`)
  A canonical, zero-mask-aware taxonomy that labels each forecast window as
  `Normal / TypeA (input-only) / TypeB (output-only) / TypeC (input+output)` based on
  per-node Q95(|dx|) thresholds estimated on the training split only.

- **Past-observable transition severity** (`calibration/severity.py`)
  A robust, train-only severity score `s = |dx - median| / (1.4826 * MAD)`, clipped at the
  training-set Q99, summarized into a per-window in-severity `r^in` that is observable at
  inference time (deployment-aware: no future information is read).

- **Severity-conditioned Mondrian conformal calibration** (`calibration/mondrian.py`)
  Hard-bin Mondrian conformal prediction (main method, retains the in-group finite-sample
  guarantee) plus a smooth interpolated variant (engineering enhancement). Includes the
  pre-registered finite-sample quantile correction `k=ceil((n+1)(1-alpha))`.

- **Evaluation metrics and protocol** (`evaluation/`)
  - `protocol.py`: a single, frozen evaluation entry point (window conventions, valid-target
    mask `true>0`, `alpha=0.10`).
  - `bootstrap.py`: window-level **paired stationary bootstrap** CIs for the method-effect
    `deltaWCE = WCE_uncond - WCE_method` and `deltaGCD`.

- **Data preprocessing** (`data/`)
  PEMS (PEMS03/04/07/08) preprocessing: raw loading, missing-value interpolation,
  train-only Z-score normalization, sliding-window segmentation, adjacency construction
  (thresholded Gaussian kernel), and time-index generation (time-of-day / day-of-week).

- **Unit tests** (`tests/unit/`) — the audit acceptance tests used to fix Round-1 issues
  (P0-2 taxonomy misalignment, P0-3 valid mask, P0-5 hard/interp separation,
  P0-6 finite-sample quantile, P0-7 bootstrap unit, P1-1 Q99 clip, P1-3 severity fallback).

## Quick start

```bash
# 1. Install the lightweight dependencies (NumPy + pandas + SciPy only for the core library)
pip install -r requirements.txt

# 2. Run the unit tests (pure NumPy, no data or GPU required)
pytest tests/unit -q
```

## Reproduce the paper tables

The experiments require the traffic forecasting point-forecast models. Please follow the
checkpoint/pretrained-model instructions below first.

```bash
# Reproduce the main calibration table (Table 4: unconditional / severity / flow-level /
# predicted-magnitude Mondrian hard + interp, with WCE / GCD / per-group width)
python experiments/reproduce_table4.py

# Reproduce the window-level paired stationary bootstrap CI (method-effect deltaWCE / deltaGCD)
python experiments/reproduce_bootstrap.py

# Reproduce the normalized-conformal control (P1-9) and the flow-stratified bootstrap (P1-7/8)
python experiments/conditioning_normalized.py
python experiments/conditioning_flowstrat.py

# Reproduce the train-inference scale equivalence gate (R3)
python experiments/reproduce_equiv.py
```

## Repository layout

```
calibration/          transition taxonomy, severity, Mondrian conformal calibration
evaluation/           frozen evaluation protocol, window-level stationary bootstrap
data/                 PEMS preprocessing + time-index generation
experiments/          reproduction scripts for the paper tables and training reference scripts
tests/unit/           audit acceptance tests (pure NumPy)
configs/              dataset configuration for the main reproduction runs
figures/              generated figures
```

## Data

Download the PEMS03/04/07/08 datasets from the official Caltrans Performance Measurement
System (PeMS) source and run:

```bash
python data/preprocess.py --dataset all
python data/generate_time_index.py --dataset all
```

See `data/README.md` for the exact split/format conventions. Raw PeMS data and model
checkpoints are **not** redistributed in this repository.

## Model checkpoints

The pretrained STAEformer / legacy backbones are not uploaded. The training scripts under
`experiments/train_*.py` are reference implementations that assume the external
[BasicTS](https://github.com/GestaltCogTeam/BasicTS) framework and GPU environment; set the
`BASICTS_ROOT` environment variable to point at the framework checkout.

## Citation

If you use this code or the ideas in the paper, please cite:

```bibtex
@article{transition_calibration,
  author  = {Wang, Ping and Zhang, Xiaoyuan},
  title   = {Beyond Marginal Coverage: Transition-Aware Uncertainty Calibration for
             Reliable Spatio-Temporal Forecasting},
  journal = {Applied Intelligence},
  year    = {2026}
}
```

## License

This repository is released under the MIT License (see `LICENSE`).
