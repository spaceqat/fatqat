"""Tests SCQubitIBMSimulator: native gate set, grid mapping, and noise model."""

import numpy as np
import pytest

from fatqat import operations as ops
from fatqat.backends import ApplyChannelStep, SCQubitIBMSimulator
from fatqat.backends.fake_superconducting import (
    fake_superconducting_ibm_implementation_map,
)
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import AmplitudeDamping, Depolarizing, PhaseDamping
from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister
from fatqat.resource_layout import ResourceLayout

# --- implementation map -----------------------------------------------------


def test_fake_superconducting_ibm_map_exposes_native_device_operands_for():
    m = fake_superconducting_ibm_implementation_map()

    assert m.supports(ops.X)
    assert m.supports(ops.SX)
    assert m.supports(ops.RZ)
    assert m.supports(ops.CZ)
    assert (0, 1) in m.device_operands_for(ops.CZ)
    assert (1, 0) in m.device_operands_for(ops.CZ)
    assert (0, 4) in m.device_operands_for(ops.CZ)
    assert (4, 0) in m.device_operands_for(ops.CZ)
    assert (0, 5) not in m.device_operands_for(ops.CZ)


def test_fake_superconducting_ibm_grid_size_controls_capacity_and_connectivity():
    backend = SCQubitIBMSimulator(grid_size=(2, 3))
    m = backend.implementation_map
    assert (2, 5) in m.device_operands_for(ops.CZ)
    assert (0, 3) in m.device_operands_for(ops.CZ)
    assert (0, 4) not in m.device_operands_for(ops.CZ)

    p = Program(7)
    with pytest.raises(BackendValidationError, match=r"at most 6.*2x3"):
        backend.run(p)


def test_fake_superconducting_ibm_map_x_sx_rz_are_uniform():
    # X/SX/RZ are legal on any of the 16 qubits, so they are registered via
    # plain one unconstrained add() rather than explicit device-operand
    # additions. supports() + empty device_operands_for() is how a compiler
    # infers "uniform, legal on any target" instead of "not supported".
    m = fake_superconducting_ibm_implementation_map()
    assert m.supports(ops.X) and not m.device_operands_for(ops.X)
    assert m.supports(ops.SX) and not m.device_operands_for(ops.SX)
    assert m.supports(ops.RZ) and not m.device_operands_for(ops.RZ)
    assert m.implementation_for(ops.X) is not None
    assert m.implementation_for(ops.SX) is not None
    assert m.implementation_for(ops.RZ) is not None


def test_fake_superconducting_ibm_map_cz_has_no_class_keyed_rule():
    # CZ must be entirely target-aware; a stray unconstrained implementation
    # would let get() silently accept non-neighbor pairs via fallback.
    m = fake_superconducting_ibm_implementation_map()
    assert m.implementation_for(ops.CZ) is None


def test_fake_superconducting_ibm_map_rejects_cx_and_iswap():
    m = fake_superconducting_ibm_implementation_map()
    assert not m.supports(ops.CX)
    assert not m.supports(ops.iSwap)


# --- native execution / shape rejection -------------------------------------


def test_fake_backend_runs_native_neighbor_operations():
    p = Program(16)
    p.add(ops.X, 0)
    p.add(ops.SX, 0)
    p.add(ops.RZ(np.pi / 3), 0)
    p.add(ops.CZ, (0, 1))

    result = (
        SCQubitIBMSimulator()
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
        SCQubitIBMSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )

    assert isinstance(excinfo.value, BackendValidationError)


def test_fake_backend_rejects_non_native_operation_families():
    # CX and iSwap belong to no/other backends; RX/RY are native only to
    # SCQubitGoogleSimulator - none of the four are legal here.
    for op, targets in (
        (ops.CX, (0, 1)),
        (ops.iSwap, (0, 1)),
        (ops.RX(0.1), 0),
        (ops.RY(0.1), 0),
    ):
        p = Program(2)
        p.add(op, targets)

        with pytest.raises(UnsupportedOperationError):
            SCQubitIBMSimulator().run(
                p, result_config={"counts": False, "final_state": True}
            )


def test_fake_backend_accepts_fewer_than_sixteen_qubits():
    p = Program(2)
    p.add(ops.SX, 0)
    p.add(ops.CZ, (0, 1))

    result = (
        SCQubitIBMSimulator()
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
    )

    state = result.get_statevector()
    assert state.shape == (2**2,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_fake_backend_rejects_more_than_sixteen_qubits():
    p = Program(17)
    p.add(ops.SX, 0)

    with pytest.raises(
        BackendValidationError, match="SCQubitIBMSimulator supports at most 16"
    ):
        SCQubitIBMSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_fake_backend_rejects_non_qubit_dimension_registers():
    p = Program([QuantumRegister(16, dim=3)])
    p.add(ops.SX, 0)

    with pytest.raises(
        BackendValidationError,
        match="SCQubitIBMSimulator only supports qubit dimensions",
    ):
        SCQubitIBMSimulator().run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_fake_backend_allows_measurement_and_reset_on_any_qubit():
    p = Program(16, 1)
    p.add(ops.Reset, 3)
    p.measure(3, 0)

    result = (
        SCQubitIBMSimulator().run(p, shots=4, result_config={"counts": True}).result()
    )

    assert result.get_counts()


# --- GridRegister-aware mapping ----------------------------------------------


def test_grid_register_binds_top_left_on_4x4_device():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    backend = SCQubitIBMSimulator()
    resource_layout = backend._resolve_resource_layout(p)
    assert isinstance(resource_layout, ResourceLayout)
    assert tuple(resource_layout.device_label(atoms[i]) for i in range(6)) == (
        0,
        1,
        2,
        4,
        5,
        6,
    )


def test_grid_register_scalar_ref_uses_grid_binder_device_label_not_identity():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    backend = SCQubitIBMSimulator()
    resource_layout = backend._resolve_resource_layout(p)
    assert resource_layout.device_label(atoms[3]) == 4


def test_rejects_grid_register_combined_with_other_quantum_register():
    atoms = GridRegister(2, 2, name="atoms")
    other = QuantumRegister(2, name="q")
    p = Program([atoms, other])
    backend = SCQubitIBMSimulator()
    with pytest.raises(BackendValidationError, match="SCQubitIBMSimulator"):
        backend._resolve_resource_layout(p)


def test_rejects_two_grid_registers():
    atoms1 = GridRegister(2, 2, name="a1")
    atoms2 = GridRegister(2, 2, name="a2")
    p = Program([atoms1, atoms2])
    backend = SCQubitIBMSimulator()
    with pytest.raises(BackendValidationError, match="SCQubitIBMSimulator"):
        backend._resolve_resource_layout(p)


def test_rejects_grid_that_does_not_fit_device_even_with_enough_total_capacity():
    atoms = GridRegister(8, 2, name="atoms")
    p = Program([atoms])
    backend = SCQubitIBMSimulator()
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p)


def test_naive_scalar_program_still_uses_declaration_order_identity_binding():
    p = Program(3)
    backend = SCQubitIBMSimulator()
    ref = p.quantum_registers[0][2]
    resource_layout = backend._resolve_resource_layout(p)
    assert resource_layout.device_label(ref) == 2


def test_grid_register_program_runs_end_to_end():
    atoms = GridRegister(1, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.SX, atoms[0])
    p.add(ops.CZ, (atoms[0], atoms[1]))
    result = (
        SCQubitIBMSimulator()
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
    )
    state = result.get_statevector()
    assert state.shape == (2**3,)
    assert np.isclose(np.linalg.norm(state), 1.0)


# --- calibration-derived default noise (from-backend flow) ------------------


def _sx_sx_program():
    program = Program(1, 1)
    program.add(ops.SX, 0)
    program.add(ops.SX, 0)
    program.measure(0, 0)
    return program


def test_backend_is_ideal_by_default():
    counts = (
        SCQubitIBMSimulator()
        .run(_sx_sx_program(), shots=100, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )

    assert counts == {"1": 100}  # SX SX = X up to phase, no noise


def test_default_noise_model_is_fully_supported():
    model = SCQubitIBMSimulator.default_noise_model()
    report = SCQubitIBMSimulator().validate_noise(model)

    assert report.supported is True
    assert set(report.accepted_sources) == {
        "AmplitudeDamping",
        "Depolarizing",
        "PhaseDamping",
        "readout_error",
    }
    assert model.channel_types() == frozenset(
        {AmplitudeDamping, Depolarizing, PhaseDamping}
    )


def test_noisy_backend_leaks_errors_but_stays_mostly_correct():
    backend = SCQubitIBMSimulator(noise=SCQubitIBMSimulator.default_noise_model())
    shots = 4000
    counts = (
        backend.run(_sx_sx_program(), shots=shots, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )

    # Readout p10 = 0.04 dominates the tiny relaxation rates.
    assert 0 < counts.get("0", 0) < 0.10 * shots


def test_x_carries_relaxation_like_sx():
    # Both X and SX are physical single-qubit pulses on this backend, so
    # both should carry the same relaxation channels - unlike RZ, which is
    # virtual. A single X flip should leak errors just like SX SX does.
    backend = SCQubitIBMSimulator(noise=SCQubitIBMSimulator.default_noise_model())
    program = Program(1, 1)
    program.add(ops.X, 0)
    program.measure(0, 0)
    counts = (
        backend.run(program, shots=4000, simulation_config={"seed": 1})
        .result()
        .get_counts()
    )

    assert 0 < counts.get("0", 0) < 0.10 * 4000


def test_rz_stays_noise_free():
    backend = SCQubitIBMSimulator(noise=SCQubitIBMSimulator.default_noise_model())
    program = Program(1)
    program.add(ops.RZ(0.7), 0)  # virtual gate: no relaxation attached
    state = (
        backend.run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.isclose(abs(state[0]), 1.0)


def test_default_noise_model_is_a_fresh_extensible_model():
    first = SCQubitIBMSimulator.default_noise_model()
    second = SCQubitIBMSimulator.default_noise_model()
    first.add_channel(ops.SX, Depolarizing(p=0.5))

    assert Depolarizing in first.channel_types()
    # Each call builds an independent model; user edits never leak back.
    program = Program(1)
    backend = SCQubitIBMSimulator()
    assert not any(
        isinstance(c, Depolarizing) and c.p == 0.5
        for c, _extent in second.channels_for(
            ops.SX,
            (program.quantum_registers[0][0],),
            backend._resolve_resource_layout(program),
        )
    )


def test_cz_has_scoped_relaxation_before_joint_depolarizing():
    program = Program(2)
    program.add(ops.CZ, (0, 1))
    backend = SCQubitIBMSimulator(noise=SCQubitIBMSimulator.default_noise_model())
    plan, _ = backend._lower_program(program)

    assert [
        step.target_indices for step in plan if isinstance(step, ApplyChannelStep)
    ] == [(0,), (0,), (1,), (1,), (0, 1)]
