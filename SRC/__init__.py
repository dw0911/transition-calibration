# Backwards-compatible alias package.
#
# The paper-release repository reorganizes the original flat `SRC/` layout into
# `calibration/` (core method) and `evaluation/` (protocol + bootstrap). To keep
# the legacy `import conformal / taxonomy / severity / evaluation_protocol /
# bootstrap` statements used by the experiments and unit tests working without
# edits, this package re-exports the relocated modules under their old names.
import os
import sys

# Ensure the repository root (containing `calibration` and `evaluation`) is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
