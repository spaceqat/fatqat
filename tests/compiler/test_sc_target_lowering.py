import numpy as np

import fatqat as fq
from fatqat.compiler.dialects import (
    NativeGate,
    NativeMeasure,
    NativeReset,
    SCNativeProgram,
)
from fatqat.compiler.dialects.sc_native import _RotationNativeProgram
from fatqat.compiler.passes import (
    lower_sc_to_native_program,
    normalize_sc_program,
    snapshot_program,
)
from fatqat.compiler.passes.sc_target import _lower_sc_to_rotation_program
from fatqat.implementation import default_matrix_implementation_map
from fatqat.operations.fixed_gates import CZGate, SXGate, XGate, iSwapGate
from fatqat.operations.parametric_gates import RX, RY, RZ
from fatqat.simulator import SCQubitSimulator
from fatqat.simulator.fake_superconducting import _SCQubitRotationSimulator

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


def test_native_lowering_emits_only_canonical_gates_on_supplied_couplings():
    backend = SCQubitSimulator(num_qubits=3, couplings=LINE)
    source = _triangle_program()

    result = lower_sc_to_native_program(source, backend, seed=7)

    assert type(result) is SCNativeProgram
    routed_gates = [
        instruction
        for instruction in result.operations
        if isinstance(instruction, NativeGate) and instruction.generated_by
    ]
    assert routed_gates
    assert all(not gate.origin_ids for gate in routed_gates)
    assert all(gate.generated_by.startswith("route.swap.") for gate in routed_gates)
    for instruction in result.operations:
        if not isinstance(instruction, NativeGate):
            continue
        assert type(instruction.operation) in (XGate, SXGate, RZ, CZGate)
        if len(instruction.sites) == 2:
            assert frozenset(instruction.sites) in {frozenset(edge) for edge in LINE}
    for layout in (result.initial_layout, result.final_layout):
        assert {qubit for qubit, _site in layout} == set(source.qubits)
        assert {site for _qubit, site in layout} <= set(backend.device_sites)


def test_rotation_lowering_emits_only_native_gates_on_supplied_couplings():
    backend = _SCQubitRotationSimulator(num_qubits=3, couplings=LINE)

    result = _lower_sc_to_rotation_program(_triangle_program(), backend, seed=7)

    assert type(result) is _RotationNativeProgram
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


def test_native_rx_decomposition_is_equivalent_up_to_global_phase():
    theta = 0.37
    source = _sc_program(1, ((fq.operations.RX(theta), 0),))
    backend = SCQubitSimulator(num_qubits=1, couplings=())

    result = lower_sc_to_native_program(source, backend, seed=3)

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
    backend = SCQubitSimulator(num_qubits=2, couplings=((0, 1),))

    result = lower_sc_to_native_program(source, backend, seed=0)

    assert result.operations
    for instruction in result.operations:
        assert isinstance(instruction, NativeGate)
        assert instruction.origin_ids == ("logical.0",)
        assert instruction.generated_by is None


def test_measurement_uses_routed_event_site_and_original_clbit():
    source = _triangle_program(measure=True)
    backend = _SCQubitRotationSimulator(num_qubits=3, couplings=LINE)

    result = _lower_sc_to_rotation_program(source, backend, seed=5)

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


def test_reset_is_lowered_with_its_semantic_origin():
    source = _sc_program(
        2,
        (
            (fq.operations.X, 0),
            (fq.operations.Reset, 0),
        ),
    )
    backend = SCQubitSimulator(num_qubits=2, couplings=((0, 1),))

    result = lower_sc_to_native_program(source, backend, seed=0)

    resets = [
        instruction
        for instruction in result.operations
        if isinstance(instruction, NativeReset)
    ]
    assert len(resets) == 1
    assert resets[0].origin_ids == ("logical.1",)
    assert resets[0].site in backend.device_sites


def test_fixed_seed_repeats_operations_and_layouts():
    source = _triangle_program(measure=True)
    backend = SCQubitSimulator(num_qubits=3, couplings=LINE)

    first = lower_sc_to_native_program(source, backend, seed=11)
    second = lower_sc_to_native_program(source, backend, seed=11)

    assert second == first
