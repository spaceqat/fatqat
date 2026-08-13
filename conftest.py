import importlib.util

import pytest

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


def pytest_addoption(parser):
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help=(
            "Exit non-zero if any test was skipped. For the CI job that "
            "installs every optional group, where a skip means a test lost "
            "its coverage rather than that a dependency is absent."
        ),
    )


def pytest_sessionfinish(session, exitstatus):
    """Turn skips into a failed run when ``--fail-on-skip`` is given.

    The job that installs every optional group has no legitimate reason to
    skip: every guard in the suite is `importorskip` on a dependency that job
    installs. Without this the invariant would only be a comment, and a test
    that quietly started skipping would look exactly like a green run.
    """
    if not session.config.getoption("--fail-on-skip"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", [])
    if not skipped:
        return
    reporter.write_sep("=", "skips are failures in this run", red=True, bold=True)
    for report in skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else ""
        reporter.write_line(f"  {report.nodeid}  {reason}")
    reporter.write_line(
        f"{len(skipped)} skipped with --fail-on-skip; every optional group is "
        "installed in this job, so nothing should be skipped."
    )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
