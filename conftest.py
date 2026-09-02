import importlib.util
import os

os.environ["MPLBACKEND"] = "Agg"

pytest_plugins = "scipy_doctest"

# Qiskit is an optional test dependency. Exclude the integration package from
# source-tree doctest collection when it is not installed.
collect_ignore = []
if importlib.util.find_spec("qiskit") is None:
    collect_ignore.append("src/fatqat/qiskit")
