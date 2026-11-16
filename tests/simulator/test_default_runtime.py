"""Public runtime defaults affect future backends without changing overrides."""

import subprocess
import sys

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError


@pytest.fixture(autouse=True)
def restore_default():
    previous = fq.simulator.get_default_runtime()
    yield
    fq.simulator.set_default_runtime(previous)


def test_default_applies_at_construction_and_explicit_runtime_wins():
    program = fq.Program(1)
    program.add(fq.operations.H, 0)

    def runtime(backend):
        return backend.run(program).result().metadata["runtime"]

    assert fq.simulator.get_default_runtime() == "numba"
    before = fq.simulator.Simulator()
    fq.simulator.set_default_runtime("NUMPY")
    assert fq.simulator.get_default_runtime() == "numpy"
    after = fq.simulator.Simulator()
    assert runtime(before) == "numba"
    assert runtime(fq.simulator.Simulator(runtime="numba")) == "numba"
    result = after.run(program).result()
    assert result.metadata["runtime"] == "numpy"
    np.testing.assert_allclose(result.get_statevector(), [2**-0.5, 2**-0.5])

    fq.simulator.set_default_runtime("numba")
    assert runtime(after) == "numpy"
    assert runtime(fq.simulator.Simulator(runtime=None)) == "numba"
    assert runtime(fq.simulator.Simulator(runtime="numpy")) == "numpy"


@pytest.mark.parametrize(
    "factory", [fq.simulator.AtomArraySimulator, fq.simulator.SCQubitSimulator]
)
def test_hardware_profiles_inherit_default(factory):
    fq.simulator.set_default_runtime("numpy")
    assert factory().run(fq.Program(1)).result().metadata["runtime"] == "numpy"


@pytest.mark.parametrize("invalid", ["jax", None])
def test_invalid_setting_preserves_previous_default(invalid):
    fq.simulator.set_default_runtime("numpy")
    with pytest.raises(BackendValidationError, match="runtime"):
        fq.simulator.set_default_runtime(invalid)
    assert fq.simulator.get_default_runtime() == "numpy"


def test_numpy_default_executes_when_numba_is_unavailable():
    code = """
import importlib.util
import sys
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: (
    None if name == 'numba' else find_spec(name, *args)
)
class BlockNumba:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'numba':
            raise ImportError('numba is unavailable')
sys.meta_path.insert(0, BlockNumba())
import fatqat as fq
from fatqat.errors import BackendValidationError
fq.simulator.set_default_runtime('numpy')
assert fq.simulator.Simulator().run(fq.Program(1)).result().metadata['runtime'] == 'numpy'
assert 'numba' not in sys.modules
try:
    fq.simulator.Simulator(runtime='numba')
except BackendValidationError:
    pass
else:
    raise AssertionError('an explicit numba request must not fall back')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
