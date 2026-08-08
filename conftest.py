import importlib.util

pytest_plugins = "scipy_doctest"

# These modules import numba at module load. numba is an optional dependency
# (the `numba` group), so exclude them from `--doctest-modules` collection when
# numba is not installed - otherwise importing them fails the run. The
# `test-numba` CI job installs numba and collects them normally.
#
# Every module that imports numba at module scope belongs in this list. The
# import being guarded at its *call site* is not enough: `--doctest-modules`
# imports the file directly, whatever its callers do.
collect_ignore = []
if importlib.util.find_spec("numba") is None:
    collect_ignore.append("src/fatqat/simulator/_engine/nb.py")
    collect_ignore.append("src/fatqat/simulator/_engine/expectation_nb.py")
    collect_ignore.append("src/fatqat/noise/nb.py")
