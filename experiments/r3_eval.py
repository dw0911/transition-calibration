# -*- coding: utf-8 -*-
"""Compatibility facade for the four-split canonical evaluation pipeline.

The original `r3_eval.py` is shipped as `reproduce_table4.py`; this module re-exports its
public API (DATASET_CFG, load_full, build_model, infer_range, and the taxonomy/severity
facades) so downstream scripts (`reproduce_bootstrap.py`, `conditioning_normalized.py`,
`conditioning_flowstrat.py`) keep working unchanged.
"""
from reproduce_table4 import (  # noqa: F401
    DATASET_CFG,
    load_full,
    build_model,
    infer_range,
    tx,
    sev,
    cf,
)
