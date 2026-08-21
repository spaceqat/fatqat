import importlib.util

pytest_plugins = "scipy_doctest"

# Qiskit is an optional test dependency. Exclude the integration package from
# doctest collection when it is not installed, because --doctest-modules would
# otherwise import it directly and fail collection.
collect_ignore = []
if importlib.util.find_spec("qiskit") is None:
    collect_ignore.append("src/fatqat/qiskit")
