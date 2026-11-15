"""Run one fixture through one framework and print a one-line outcome.

    python -m nirconf.probe <framework> <suite> <fixture>

Used by check NT-3 to compare outcomes across PYTHONHASHSEED values.
"""

import contextlib
import hashlib
import io
import logging
import sys
import warnings
from pathlib import Path

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import nir  # noqa: E402
import numpy as np  # noqa: E402

from . import adapters  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures"

if __name__ == "__main__":
    fw, suite, name = sys.argv[1:4]
    d = FIX / suite / name
    try:
        g = nir.read(d / "graph.nir")
    except Exception:  # noqa: BLE001
        g = nir.read(d / "graph_legacy.nir")
    x = np.load(d / "input.npy")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            y = adapters.run(fw, g, x, 1.0 if suite == "dt1" else 1e-4)
        print("ok", hashlib.sha1(np.ascontiguousarray(y).tobytes()).hexdigest()[:10])
    except Exception as e:  # noqa: BLE001
        print("error", type(e).__name__)
