import numpy as np

import fatqat as fq
from fatqat.compiler.dialects import NativeGate, NativeMeasure
from fatqat.compiler.passes import (
    lower_sc_to_google_program,
    lower_sc_to_ibm_program,
    normalize_sc_program,
    snapshot_program,
)
from fatqat.implementation import default_matrix_implementation_map
from fatqat.operations.fixed_gates import CZGate, SXGate, XGate, iSwapGate
from fatqat.operations.parametric_gates import RX, RY, RZ
from fatqat.simulator import SCQubitGoogleSimulator, SCQubitIBMSimulator

LINE = ((0, 1), (1, 2))


def _sc_program(num_qubits: int, gates, *, measure: bool = False):
    program = fq.Program(num_qubits, num_qubits if measure else 0)
    for operation, targets in gates:
        program.add(operation, targets)
    if measure:
        program.measure_all()
    return normalize_sc_program(snapshot_program(program))


def _triangle_program(*, measure: bool = False):
    return _sc_program(
        3,
        (
            (fq.operations.CZ, (0, 1)),
            (fq.operations.CZ, (1, 2)),
            (fq.operations.CZ, (0, 2)),
        ),
        measure=measure,
    )


def test_ibm_lowering_emits_only_native_gates_on_supplied_couplings():
    backend = SCQubitIBMSimulator(num_qubits=3, couplings=LINE)

    result = lower_sc_to_ibm_program(_triangle_program(), backend, seed=7)

    assert any(
        isinstance(instruction, NativeGate) and instruction.generated_by
        for instruction in result.operations
    )
    for instruction in result.operations:
        if not isinstance(instruction, NativeGate):
            continue
        assert type(instruction.operation) in (XGate, SXGate, RZ, CZGate)
        if len(instruction.sites) == 2:
            assert frozenset(instruction.sites) in {frozenset(edge) for edge in LINE}


def test_google_lowering_emits_only_native_gates_on_supplied_couplings():
    backend = SCQubitGoogleSimulator(num_qubits=3, couplings=LINE)

    result = lower_sc_to_google_program(_triangle_program(), backend, seed=7)

    assert any(
        isinstance(instruction, NativeGate) and instruction.generated_by
        for instruction in result.operations
    )
    for instruction in result.operations:
        if not isinstance(instruction, NativeGate):
            continue
        assert type(instruction.operation) in (RX, RY, RZ, iSwapGate, CZGate)
        if len(instruction.sites) == 2:
            assert frozenset(instruction.sites) in {frozenset(edge) for edge in LINE}


def test_ibm_rx_decomposition_is_equivalent_up_to_global_phase():
    theta = 0.37
    source = _sc_program(1, ((fq.operations.RX(theta), 0),))
    backend = SCQubitIBMSimulator(num_qubits=1, couplings=())

    result = lower_sc_to_ibm_program(source, backend, seed=3)

    gates = [
        instruction.operation
        for instruction in result.operations
        if isinstance(instruction, NativeGate)
    ]
    qref = fq.QuantumRegister(1, name="q")[0]
    implementation_map = default_matrix_implementation_map()
    actual = np.eye(2, dtype=complex)
    for gate in gates:
        rule = implementation_map.implementation_for(gate)
        actual = rule(gate, targets=(qref,)) @ actual
    expected_rule = implementation_map.implementation_for(fq.operations.RX(theta))
    expected = expected_rule(fq.operations.RX(theta), targets=(qref,))
    phase = np.vdot(expected.ravel(), actual.ravel()) / 2

    assert np.allclose(actual, phase / abs(phase) * expected)


def test_semantic_swap_decomposition_keeps_origins_not_route_provenance():
    source = _sc_program(2, ((fq.operations.Swap, (0, 1)),))
    backend = SCQubitIBMSimulator(num_qubits=2, couplings=((0, 1),))

    result = lower_sc_to_ibm_program(source, backend, seed=0)

    assert result.operations
    for instruction in result.operations:
        assert isinstance(instruction, NativeGate)
        assert instruction.origin_ids == ("logical.0",)
        assert instruction.generated_by is None


def test_measurement_uses_routed_event_site_and_original_clbit():
    source = _triangle_program(measure=True)
    backend = SCQubitGoogleSimulator(num_qubits=3, couplings=LINE)

    result = lower_sc_to_google_program(source, backend, seed=5)

    measurements = [
        instruction
        for instruction in result.operations
        if isinstance(instruction, NativeMeasure)
    ]
    final_layout = dict(result.final_layout)
    assert len(measurements) == 3
    for measurement in measurements:
        logical = next(
            node.qubits[0]
            for node in source.nodes
            if node.origin_ids == measurement.origin_ids
        )
        assert measurement.site == final_layout[logical]
        assert measurement.clbit in source.clbits
