"""The SC adapter evolves private qutrit state without exposing Qobj values."""

import json
from pathlib import Path

import numpy as np

import fatqat as fq
from fatqat.backends.pulse.backend import PulseBackend
from fatqat.backends.pulse.execution import execute_with_boundaries
from fatqat.backends.pulse.qutip_adapter import SCQutipAdapter
from fatqat.backends.pulse.superconducting import (
    load_calibration_spec,
    load_physics_model,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _backend():
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return PulseBackend(model, calibration)


def test_adapter_returns_a_normalized_full_qutrit_density_matrix():
    backend = _backend()
    program = fq.Program(1)
    program.add(fq.ops.RX(np.pi), 0)
    plan, _ = backend._lower_program(program)
    adapter = SCQutipAdapter(backend.model)
    execute_with_boundaries(plan, adapter.evolve, lambda *_: None)
    density = adapter.density_matrix()

    assert density.shape == (9, 9)
    assert np.isclose(np.trace(density), 1.0)
    assert 1.0 - density[0, 0].real > 0.5


def test_backend_exposes_numpy_density_matrix_for_a_continuous_program():
    backend = _backend()
    program = fq.Program(1)
    program.add(fq.ops.RX(0.4), 0)
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()

    density = result.get_density_matrix()
    assert density.shape == (9, 9)
    assert result.available_data == frozenset({"density_matrix"})
