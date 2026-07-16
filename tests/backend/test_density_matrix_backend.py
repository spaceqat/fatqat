"""Tests density-matrix backend public behavior: counts, state availability, validation."""

import warnings

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.errors import (
    BackendValidationError,
    NoMeasurementWarning,
    ResultFieldUnavailableError,
)
from fatqat import operations as ops
from fatqat.program import Program


def test_counts_default_with_measurement():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    result = SimulatorBackend("DM").run(p, shots=10, seed=0).result()
    assert result.get_counts() == {"1": 10}


def test_density_matrix_default_attached_when_no_measurement():
    p = Program(1)
    p.add(ops.H, 0)
    job = SimulatorBackend("DM").run(p, result_config={"counts": False})
    rho = job.result().get_density_matrix()
    assert np.allclose(rho, np.full((2, 2), 0.5))


def test_density_matrix_not_attached_by_default_with_measurement():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    result = SimulatorBackend("DM").run(p, shots=10, seed=0).result()
    with pytest.raises(ResultFieldUnavailableError):
        result.get_density_matrix()


def test_statevector_accessor_unavailable_on_density_matrix_result():
    p = Program(1)
    p.add(ops.H, 0)
    result = SimulatorBackend("DM").run(p, result_config={"counts": False}).result()
    with pytest.raises(ResultFieldUnavailableError):
        result.get_statevector()


def test_density_matrix_default_survives_reset():
    # Unlike the statevector backend, reset is deterministic here, so a
    # reset-bearing measurement-free program still exports its exact state.
    p = Program(1)
    p.add(ops.H, 0)
    p.add(ops.Reset, 0)
    result = SimulatorBackend("DM").run(p, result_config={"counts": False}).result()
    rho = result.get_density_matrix()
    assert np.allclose(rho, [[1, 0], [0, 0]])


def test_reset_of_entangled_qubit_gives_exact_mixed_state():
    p = Program(2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.add(ops.Reset, 0)
    result = SimulatorBackend("DM").run(p, result_config={"counts": False}).result()
    rho = result.get_density_matrix()
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 0.5
    expected[2, 2] = 0.5
    assert np.allclose(rho, expected)


def test_reset_program_counts_are_deterministic_for_any_shots():
    # Reset-only stochasticity does not exist on this backend, so counts on a
    # reset-then-measure program need no per-shot replay to be exact.
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add(ops.Reset, 0)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    result = SimulatorBackend("DM").run(p, shots=50, seed=3).result()
    assert result.get_counts() == {"1": 50}


def test_counts_distribution_matches_born_rule():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    counts = SimulatorBackend("DM").run(p, shots=2000, seed=11).result().get_counts()
    assert set(counts) == {"0", "1"}
    assert abs(counts["0"] - 1000) < 150


def test_density_matrix_with_measurement_shots_one():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    result = (
        SimulatorBackend("DM")
        .run(p, shots=1, seed=0, result_config={"counts": True, "density_matrix": True})
        .result()
    )
    rho = result.get_density_matrix()
    # collapsed to a basis state: pure, single nonzero diagonal entry
    assert np.isclose(np.real(np.trace(rho)), 1.0)
    assert np.count_nonzero(np.round(np.abs(np.diag(rho)), 6)) == 1
    (key,) = result.get_counts()
    assert np.isclose(np.real(rho[int(key), int(key)]), 1.0)


def test_density_matrix_with_measurement_and_many_shots_rejected():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    with pytest.raises(BackendValidationError):
        SimulatorBackend("DM").run(
            p, shots=10, result_config={"counts": True, "density_matrix": True}
        )


def test_counts_require_positive_shots():
    p = Program(1, 1)
    p.add_measurement(0, 0)
    with pytest.raises(BackendValidationError):
        SimulatorBackend("DM").run(p, shots=0)


def test_feedforward_condition_applies():
    p = Program(2, 2)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 1, condition=(0, 1))
    p.add_measurement(1, 1)
    counts = SimulatorBackend("DM").run(p, shots=20, seed=0).result().get_counts()
    assert counts == {"11": 20}


def test_dynamic_counts_deterministic_for_fixed_seed():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 1, condition=(0, 1))
    p.add_measurement(1, 1)
    backend = SimulatorBackend("DM")
    first = backend.run(p, shots=64, seed=42).result().get_counts()
    second = backend.run(p, shots=64, seed=42).result().get_counts()
    assert first == second
    # feedforward correlates the two clbits: only 00 and 11 can appear
    assert set(first) <= {"00", "11"}


def test_dynamic_counts_parallel_matches_serial():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 1, condition=(0, 1))
    p.add_measurement(1, 1)
    serial = (
        SimulatorBackend("DM", {"parallel_mode": "serial"})
        .run(p, shots=40, seed=9)
        .result()
        .get_counts()
    )
    parallel = (
        SimulatorBackend("DM", {"max_workers": 2})
        .run(p, shots=40, seed=9)
        .result()
        .get_counts()
    )
    assert serial == parallel


def test_diagonal_matches_statevector_probabilities():
    # Backend-neutral cross-check: for a pure circuit, diag(rho) equals the
    # statevector's Born probabilities.
    p = Program(2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.add(ops.RY(0.7), 1)
    rho = (
        SimulatorBackend("DM")
        .run(p, result_config={"counts": False})
        .result()
        .get_density_matrix()
    )
    sv = (
        fq.backends.SimulatorBackend("SV")
        .run(p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(np.real(np.diag(rho)), np.abs(sv) ** 2)
    assert np.allclose(rho, np.outer(sv, sv.conj()))


def test_result_metadata_records_backend_shots_and_config():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    config = {"counts": True, "density_matrix": False}
    result = (
        SimulatorBackend("DM").run(p, shots=7, seed=0, result_config=config).result()
    )
    assert result.metadata["shots"] == 7
    assert result.metadata["backend_name"] == "SimulatorBackend"
    assert result.metadata["result_config"] == config


def test_run_warns_and_ignores_unknown_result_config_keys():
    p = Program(1)
    p.add(ops.H, 0)
    with pytest.warns(
        UserWarning, match="SimulatorBackend ignored unsupported result_config options"
    ):
        result = (
            SimulatorBackend("DM")
            .run(p, result_config={"counts": False, "statevector": True})
            .result()
        )
    assert result.metadata["result_config"] == {
        "counts": False,
        "density_matrix": None,
    }


def test_backend_warns_and_ignores_unknown_options():
    with pytest.warns(
        UserWarning, match="SimulatorBackend ignored unsupported backend options"
    ):
        SimulatorBackend("DM", {"gpu": True})


def test_run_rejects_non_dict_result_config():
    p = Program(1)
    p.add(ops.H, 0)
    with pytest.raises(TypeError, match="dict or None"):
        SimulatorBackend("DM").run(p, result_config=object())


def test_no_measurement_warning_when_counts_only_and_no_state():
    p = Program(1, 1)  # has a clbit, never measured
    p.add(ops.H, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimulatorBackend("DM").run(
            p,
            shots=10,
            seed=0,
            result_config={"counts": True, "density_matrix": False},
        ).result()
    assert any(issubclass(w.category, NoMeasurementWarning) for w in caught)


def test_qutrit_program_counts_and_state():
    qt = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qt])
    program.add(fq.ops.Shift(1), qt[0])
    rho = (
        SimulatorBackend("DM")
        .run(program, result_config={"counts": False})
        .result()
        .get_density_matrix()
    )
    expected = np.zeros((3, 3), dtype=complex)
    expected[1, 1] = 1.0
    assert np.allclose(rho, expected)


def test_dim2_gate_on_qutrit_raises_at_lowering_not_frontend():
    qt = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qt])
    program.add(fq.ops.H, qt[0])  # frontend must NOT raise here
    with pytest.raises(BackendValidationError) as exc:
        fq.backends.SimulatorBackend("DM").run(
            program, result_config={"counts": False, "density_matrix": True}
        ).result()
    msg = str(exc.value)
    assert "H" in msg and "3" in msg  # names the op and the target dimension


def test_unsupported_operation_raises():
    class Bogus(fq.ops.Operation):
        pass

    p = Program(1)
    p.add(Bogus(), 0)
    with pytest.raises(fq.errors.UnsupportedOperationError):
        SimulatorBackend("DM").run(p, result_config={"counts": False})


# --- target-aware resolution (mirrors SimulatorBackend, see test_resolution.py) --


def test_target_aware_map_allows_registered_target_key():
    cz_rule = fq.implementation.default_matrix_implementation_map().implementation_for(
        ops.CZ
    )
    m = fq.implementation.ImplementationMap()
    m.add(ops.CZ, cz_rule, device_operands=(0, 1))
    backend = SimulatorBackend("DM", implementation_map=m)

    p = Program(2)
    p.add(ops.CZ, (0, 1))

    result = backend.run(
        p, result_config={"counts": False, "density_matrix": True}
    ).result()
    assert result.get_density_matrix().shape == (4, 4)


def test_target_aware_map_rejects_illegal_target_key():
    cz_rule = fq.implementation.default_matrix_implementation_map().implementation_for(
        ops.CZ
    )
    m = fq.implementation.ImplementationMap()
    m.add(ops.CZ, cz_rule, device_operands=(0, 1))
    backend = SimulatorBackend("DM", implementation_map=m)

    p = Program(2)
    p.add(ops.CZ, (1, 0))

    # Same UnsupportedOperationError type as an unsupported family (see
    # test below); only the message distinguishes "illegal target" from
    # "no rule at all."
    with pytest.raises(
        fq.errors.UnsupportedOperationError, match="device operands"
    ) as excinfo:
        backend.run(p, result_config={"counts": False, "density_matrix": True})

    assert isinstance(excinfo.value, BackendValidationError)


def test_target_aware_map_unsupported_family_still_raises_unsupported_operation():
    cz_rule = fq.implementation.default_matrix_implementation_map().implementation_for(
        ops.CZ
    )
    m = fq.implementation.ImplementationMap()
    m.add(ops.CZ, cz_rule, device_operands=(0, 1))
    backend = SimulatorBackend("DM", implementation_map=m)

    p = Program(1)
    p.add(ops.X, 0)

    with pytest.raises(fq.errors.UnsupportedOperationError):
        backend.run(p, result_config={"counts": False, "density_matrix": True})
