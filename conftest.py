import importlib.util

pytest_plugins = "scipy_doctest"

# `fatqat.simulator.nb` imports numba at module load. numba is an optional
# dependency (the `numba` group), so exclude it from `--doctest-modules`
# collection when numba is not installed - otherwise importing it fails the run.
# The `test-numba` CI job installs numba and collects the module normally.
collect_ignore = []
if importlib.util.find_spec("numba") is None:
    collect_ignore.append("src/fatqat/simulator/nb.py")
    collect_ignore.append("src/fatqat/noise/nb.py")
