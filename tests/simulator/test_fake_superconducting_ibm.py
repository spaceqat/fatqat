"""Tests SCQubitIBMSimulator: native gates, arbitrary topology, and noise."""

import numpy as np
import pytest

import fatqat.operations as ops
from fatqat._backends.steps import ApplyChannelStep
from fatqat.simulator import SCQubitIBMSimulator
from fatqat.simulator.fake_superconducting import (
    fake_superconducting_ibm_implementation_map,
)
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import AmplitudeDamping, Depolarizing, PhaseDamping
from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister
from fatqat.resource_layout import ResourceLayout

COUPLINGS = ((0, 1), (1, 2), (1, 3), (3, 4))

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


def test_fake_superconducting_ibm_accepts_arbitrary_couplings_and_exposes_sites():
    backend = SCQubitIBMSimulator(num_qubits=5, couplings=COUPLINGS)
    m = backend.implementation_map
    assert backend.device_sites == (0, 1, 2, 3, 4)
    assert (1, 3) in m.device_operands_for(ops.CZ)
    assert (3, 1) in m.device_operands_for(ops.CZ)
    assert (0, 4) not in m.device_operands_for(ops.CZ)

    p = Program(6)
    with pytest.raises(BackendValidationError, match=r"at most 5"):
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


# --- register mapping --------------------------------------------------------


def test_grid_register_is_flattened_on_arbitrary_topology():
    qubits = GridRegister(2, 3, name="qubits")
    p = Program([qubits])
    backend = SCQubitIBMSimulator(num_qubits=6, couplings=((0, 1),))
    resource_layout = backend._resolve_resource_layout(p)
    assert isinstance(resource_layout, ResourceLayout)
    assert tuple(resource_layout.device_label(qubits[i]) for i in range(6)) == tuple(
        range(6)
    )


def test_grid_register_can_be_combined_with_other_quantum_register():
    qubits = GridRegister(2, 2, name="qubits")
    other = QuantumRegister(2, name="q")
    p = Program([qubits, other])
    backend = SCQubitIBMSimulator(num_qubits=6, couplings=((0, 1),))
    layout = backend._resolve_resource_layout(p)
    assert tuple(layout.device_label(qubits[i]) for i in range(4)) == (0, 1, 2, 3)
    assert tuple(layout.device_label(other[i]) for i in range(2)) == (4, 5)


def test_naive_scalar_program_still_uses_declaration_order_identity_binding():
    p = Program(3)
    backend = SCQubitIBMSimulator()
    ref = p.quantum_registers[0][2]
    resource_layout = backend._resolve_resource_layout(p)
    assert resource_layout.device_label(ref) == 2


def test_grid_register_program_runs_end_to_end():
    qubits = GridRegister(1, 3, name="qubits")
    p = Program([qubits])
    p.add(ops.SX, qubits[0])
    p.add(ops.CZ, (qubits[0], qubits[1]))
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


def test_default_noise_model_is_supported_and_calibrated():
    model = SCQubitIBMSimulator.default_noise_model()
    sources = model._noise_sources()
    assert SCQubitIBMSimulator().validate_noise_model(model) is None
    assert {type(source) for source, _operation in sources} == {
        AmplitudeDamping,
        Depolarizing,
        PhaseDamping,
    }

    x_channels = {
        type(source): source
        for source, operation in sources
        if operation is type(ops.X)
    }
    t1, t2, duration = 60e-6, 48e-6, 20e-9
    assert (
        x_channels[AmplitudeDamping].p[0],
        x_channels[PhaseDamping].p,
    ) == pytest.approx(
        (
            1 - np.exp(-duration / t1),
            1 - np.exp(-(1 / t2 - 1 / (2 * t1)) * duration),
        )
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
    first.add(Depolarizing(p=0.5), operation=ops.SX)

    assert Depolarizing in {
        type(source) for source, _operation in first._noise_sources()
    }
    # Each call builds an independent model; user edits never leak back.
    program = Program(1)
    backend = SCQubitIBMSimulator()
    assert not any(
        isinstance(c, Depolarizing) and c.p == 0.5
        for c, _extent in second._noise_for_occurrence(
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
