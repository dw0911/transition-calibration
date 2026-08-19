# -*- coding: utf-8 -*-
"""Pytest configuration: make the repository root (with calibration/, evaluation/, SRC/)
importable for the unit tests, and disable bytecode caching inside the repo."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.dont_write_bytecode = True
