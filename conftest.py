import importlib.util

pytest_plugins = "scipy_doctest"

# The Numba engines are optional features whose modules import numba at
# import time. When the optional `numba` dependency group is not installed,
# skip collecting those modules so `pytest --doctest-modules src/fatqat`
# (which imports every module) does not error. Their own tests already
# `importorskip`.
collect_ignore: list[str] = []
if importlib.util.find_spec("numba") is None:
    collect_ignore.append("src/fatqat/backends/statevector_numba.py")
    collect_ignore.append("src/fatqat/backends/numba_engine.py")
