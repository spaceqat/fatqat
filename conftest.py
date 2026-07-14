import importlib.util

pytest_plugins = "scipy_doctest"

# The Numba statevector engine is an optional feature whose module imports numba
# at import time. When the optional `numba` dependency group is not installed,
# skip collecting that module so `pytest --doctest-modules src/fatqat` (which
# imports every module) does not error. Its own tests already `importorskip`.
collect_ignore: list[str] = []
if importlib.util.find_spec("numba") is None:
    collect_ignore.append("src/fatqat/backends/statevector_numba.py")
