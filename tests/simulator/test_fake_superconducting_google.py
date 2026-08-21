"""Tests SCQubitGoogleSimulator: native gate set, grid mapping, and noise model."""

import numpy as np
import pytest

from fatqat import operations as ops
from fatqat._backends.steps import ApplyChannelStep
from fatqat.simulator import SCQubitGoogleSimulator
from fatqat.simulator.fake_superconducting import (
    fake_superconducting_google_implementation_map,
)
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import AmplitudeDamping, Depolarizing, PhaseDamping
from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister
from fatqat.resource_layout import ResourceLayout

# --- implementation map -----------------------------------------------------


def test_fake_superconducting_google_map_exposes_native_device_operands_for():
    m = fake_superconducting_google_implementation_map()

    assert m.supports(ops.RX)
    assert m.supports(ops.RY)
    assert m.supports(ops.RZ)
    assert m.supports(ops.iSwap)
    assert m.supports(ops.CZ)
    for op in (ops.iSwap, ops.CZ):
        assert (0, 1) in m.device_operands_for(op)
        assert (1, 0) in m.device_operands_for(op)
        assert (0, 4) in m.device_operands_for(op)
        assert (4, 0) in m.device_operands_for(op)
        assert (0, 5) not in m.device_operands_for(op)


def test_fake_superconducting_google_grid_size_controls_grid_binding():
    qubits = GridRegister(2, 2, name="qubits")
    layout = SCQubitGoogleSimulator(grid_size=(2, 3))._resolve_resource_layout(
        Program([qubits])
    )
    assert tuple(layout.device_label(qubits[i]) for i in range(4)) == (0, 1, 3, 4)


def test_fake_superconducting_google_map_rx_ry_rz_are_uniform():
    m = fake_superconducting_google_implementation_map()
    assert m.supports(ops.RX) and not m.device_operands_for(ops.RX)
    assert m.supports(ops.RY) and not m.device_operands_for(ops.RY)
    assert m.supports(ops.RZ) and not m.device_operands_for(ops.RZ)
    assert m.implementation_for(ops.RX) is not None
    assert m.implementation_for(ops.RY) is not None
    assert m.implementation_for(ops.RZ) is not None


def test_fake_superconducting_google_map_iswap_and_cz_have_no_class_keyed_rule():
    # Both two-qubit gates must be entirely target-aware; a stray
    # unconstrained implementation would let get() silently accept
    # non-neighbor pairs via fallback.
    m = fake_superconducting_google_implementation_map()
    assert m.implementation_for(ops.iSwap) is None
    assert m.implementation_for(ops.CZ) is None


def test_fake_superconducting_google_map_rejects_cx_and_sx():
    m = fake_superconducting_google_implementation_map()
    assert not m.supports(ops.CX)
    assert not m.supports(ops.SX)
    assert not m.supports(ops.X)


# --- native execution / shape rejection -------------------------------------


def test_fake_backend_runs_native_neighbor_operations():
    p = Program(16)
    p.add(ops.RX(0.3), 0)
    p.add(ops.RY(0.3), 0)
    p.add(ops.RZ(np.pi / 3), 0)
    p.add(ops.iSwap, (0, 1))
    p.add(ops.CZ, (0, 1))

    result = (
        SCQubitGoogleSimulator()
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
    )

    state = result.get_statevector()
    assert state.shape == (2**16,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_fake_backend_rejects_non_neighbor_cz():
    p = Program(16)
    p.add(ops.CZ, (0, 5))

    with pytest.raises(UnsupportedOperationError, match="device operands") as excinfo:
        SCQubitGoogleSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )

    assert isinstance(excinfo.value, BackendValidationError)


def test_fake_backend_rejects_non_neighbor_iswap():
    p = Program(16)
    p.add(ops.iSwap, (0, 5))

    with pytest.raises(UnsupportedOperationError, match="device operands") as excinfo:
        SCQubitGoogleSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )

    assert isinstance(excinfo.value, BackendValidationError)


def test_fake_backend_rejects_non_native_operation_families():
    # CX belongs to no backend here; SX and X are native only to
    # SCQubitIBMSimulator - none of the three are legal here.
    for op, targets in ((ops.CX, (0, 1)), (ops.SX, 0), (ops.X, 0)):
        p = Program(2)
        p.add(op, targets)

        with pytest.raises(UnsupportedOperationError):
            SCQubitGoogleSimulator().run(
                p, result_config={"counts": False, "final_state": True}
            )


def test_fake_backend_accepts_fewer_than_sixteen_qubits():
    p = Program(2)
    p.add(ops.RX(0.3), 0)
    p.add(ops.CZ, (0, 1))

    result = (
        SCQubitGoogleSimulator()
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
    )

    state = result.get_statevector()
    assert state.shape == (2**2,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_fake_backend_rejects_more_than_sixteen_qubits():
    p = Program(17)
    p.add(ops.RX(0.3), 0)

    with pytest.raises(
        BackendValidationError, match="SCQubitGoogleSimulator supports at most 16"
    ):
        SCQubitGoogleSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_fake_backend_rejects_non_qubit_dimension_registers():
    p = Program([QuantumRegister(16, dim=3)])
    p.add(ops.RX(0.3), 0)

    with pytest.raises(
        BackendValidationError,
        match="SCQubitGoogleSimulator only supports qubit dimensions",
    ):
        SCQubitGoogleSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_fake_backend_allows_measurement_and_reset_on_any_qubit():
    p = Program(16, 1)
    p.add(ops.Reset, 3)
    p.measure(3, 0)

    result = (
        SCQubitGoogleSimulator()
        .run(p, shots=4, result_config={"counts": True})
        .result()
    )

    assert result.get_counts()


# --- GridRegister-aware mapping ----------------------------------------------


def test_grid_register_binds_top_left_on_4x4_device():
    qubits = GridRegister(2, 3, name="qubits")
    p = Program([qubits])
    backend = SCQubitGoogleSimulator()
    resource_layout = backend._resolve_resource_layout(p)
    assert isinstance(resource_layout, ResourceLayout)
    assert tuple(resource_layout.device_label(qubits[i]) for i in range(6)) == (
        0,
        1,
        2,
        4,
        5,
        6,
    )


def test_grid_register_scalar_ref_uses_grid_binder_device_label_not_identity():
    qubits = GridRegister(2, 3, name="qubits")
    p = Program([qubits])
    backend = SCQubitGoogleSimulator()
    resource_layout = backend._resolve_resource_layout(p)
    assert resource_layout.device_label(qubits[3]) == 4


def test_rejects_grid_register_combined_with_other_quantum_register():
    qubits = GridRegister(2, 2, name="qubits")
    other = QuantumRegister(2, name="q")
    p = Program([qubits, other])
    backend = SCQubitGoogleSimulator()
    with pytest.raises(BackendValidationError, match="SCQubitGoogleSimulator"):
        backend._resolve_resource_layout(p)


def test_rejects_two_grid_registers():
    atoms1 = GridRegister(2, 2, name="a1")
    atoms2 = GridRegister(2, 2, name="a2")
    p = Program([atoms1, atoms2])
    backend = SCQubitGoogleSimulator()
    with pytest.raises(BackendValidationError, match="SCQubitGoogleSimulator"):
        backend._resolve_resource_layout(p)


def test_rejects_grid_that_does_not_fit_device_even_with_enough_total_capacity():
    qubits = GridRegister(8, 2, name="qubits")
    p = Program([qubits])
    backend = SCQubitGoogleSimulator()
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p)


def test_naive_scalar_program_still_uses_declaration_order_identity_binding():
    p = Program(3)
    backend = SCQubitGoogleSimulator()
    ref = p.quantum_registers[0][2]
    resource_layout = backend._resolve_resource_layout(p)
    assert resource_layout.device_label(ref) == 2


def test_grid_register_program_runs_end_to_end():
    qubits = GridRegister(1, 3, name="qubits")
    p = Program([qubits])
    p.add(ops.RX(0.3), qubits[0])
    p.add(ops.iSwap, (qubits[0], qubits[1]))
    result = (
        SCQubitGoogleSimulator()
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
    )
    state = result.get_statevector()
    assert state.shape == (2**3,)
    assert np.isclose(np.linalg.norm(state), 1.0)


# --- calibration-derived default noise (from-backend flow) ------------------


def _rx_pi_program():
    program = Program(1, 1)
    program.add(ops.RX(np.pi), 0)  # RX(pi) = X, up to a phase
    program.measure(0, 0)
    return program


def test_backend_is_ideal_by_default():
    counts = (
        SCQubitGoogleSimulator()
        .run(_rx_pi_program(), shots=100, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )

    assert counts == {"1": 100}


def test_default_noise_model_is_fully_supported():
    model = SCQubitGoogleSimulator.default_noise_model()
    report = SCQubitGoogleSimulator().check_noise_support(model)

    assert report.supported is True
    assert set(report.accepted_sources) == {
        "AmplitudeDamping(p)",
        "Depolarizing(p)",
        "PhaseDamping(p)",
        "ReadoutConfusion",
    }
    assert {type(source) for source, _operation in model._noise_sources()} == {
        AmplitudeDamping,
        Depolarizing,
        PhaseDamping,
    }


def test_noisy_backend_leaks_errors_but_stays_mostly_correct():
    backend = SCQubitGoogleSimulator(noise=SCQubitGoogleSimulator.default_noise_model())
    shots = 4000
    counts = (
        backend.run(_rx_pi_program(), shots=shots, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )

    # Readout p10 = 0.04 dominates the tiny relaxation rates.
    assert 0 < counts.get("0", 0) < 0.10 * shots


def test_ry_carries_relaxation_like_rx():
    # RX, RY, and RZ are all physical single-qubit rotations on this
    # backend, so all three carry the same relaxation channels.
    backend = SCQubitGoogleSimulator(noise=SCQubitGoogleSimulator.default_noise_model())
    program = Program(1, 1)
    program.add(ops.RY(np.pi), 0)
    program.measure(0, 0)
    counts = (
        backend.run(program, shots=4000, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )

    assert 0 < counts.get("0", 0) < 0.10 * 4000


def test_rz_carries_relaxation_like_other_google_rotations():
    backend = SCQubitGoogleSimulator(noise=SCQubitGoogleSimulator.default_noise_model())
    program = Program(1)
    program.add(ops.RZ(0.7), 0)
    plan, _ = backend._lower_program(program)

    assert [
        step.target_indices for step in plan if isinstance(step, ApplyChannelStep)
    ] == [(0,), (0,)]


def test_default_noise_model_is_a_fresh_extensible_model():
    first = SCQubitGoogleSimulator.default_noise_model()
    second = SCQubitGoogleSimulator.default_noise_model()
    first.add(Depolarizing(p=0.5), operation=ops.RX)

    assert Depolarizing in {
        type(source) for source, _operation in first._noise_sources()
    }
    # Each call builds an independent model; user edits never leak back.
    program = Program(1)
    backend = SCQubitGoogleSimulator()
    assert not any(
        isinstance(c, Depolarizing) and c.p == 0.5
        for c, _extent in second._noise_for_occurrence(
            ops.RX,
            (program.quantum_registers[0][0],),
            backend._resolve_resource_layout(program),
        )
    )


def test_iswap_has_scoped_relaxation_before_joint_depolarizing():
    program = Program(2)
    program.add(ops.iSwap, (0, 1))
    backend = SCQubitGoogleSimulator(noise=SCQubitGoogleSimulator.default_noise_model())
    plan, _ = backend._lower_program(program)

    assert [
        step.target_indices for step in plan if isinstance(step, ApplyChannelStep)
    ] == [(0,), (0,), (1,), (1,), (0, 1)]
