"""Gate fusion keeps its shared gate-matrix facts usable without Numba."""

import subprocess
import sys


def test_neutral_gate_matrices_and_numpy_engine_import_without_numba():
    # NumPy engines and plan passes read structure codes and kernel specs from
    # the neutral module. A Numba import there breaks installs without the
    # extra, so the check runs in a fresh process with Numba blocked.
    code = """
import importlib
import sys
sys.modules['numba'] = None
for name in (
    'fatqat.simulator._engine.numerics._gate_matrices',
    'fatqat.simulator._engine.engines.numpy_engine',
    'fatqat.simulator._engine.engines.base',
    'fatqat',
):
    importlib.import_module(name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
