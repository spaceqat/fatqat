import numpy as np
import pytest

import fatqat as fq
from fatqat.compiler.dialects.sc_native import NativeGate, NativeMeasure
from fatqat.compiler.passes import normalize_sc_program, snapshot_program
from fatqat.compiler.passes.lq_target import lower_sc_to_lq_program
from fatqat.compiler.targets import _SCTarget


def test_lq_target_routes_to_cz_edges_and_preserves_measurement_destinations():
    source = fq.Program(3, 4)
    for pair in ((0, 1), (1, 2), (0, 2)):
        source.add(fq.operations.CZ, pair)
    source.measure(0, 3)
    source.measure(2, 0)
    target = _SCTarget("test", (3, 7, 9), frozenset({(3, 7), (7, 9)}), "lqcloud")
    result = lower_sc_to_lq_program(
        normalize_sc_program(snapshot_program(source)), target, seed=7
    )
    layout = dict(result.final_layout)
    measurements = [op for op in result.operations if isinstance(op, NativeMeasure)]
    assert [(op.site, op.clbit) for op in measurements] == [
        (layout[source.quantum_registers[0][0]], source.classical_registers[0][3]),
        (layout[source.quantum_registers[0][2]], source.classical_registers[0][0]),
    ]
    gates = [op for op in result.operations if isinstance(op, NativeGate)]
    assert any(op.generated_by for op in gates)
    for op in gates:
        assert op.operation.name in ("H", "RZ", "CZ")
        if op.operation is fq.operations.CZ:
            assert frozenset(op.sites) in {frozenset((3, 7)), frozenset((7, 9))}


@pytest.mark.parametrize("gate", [fq.operations.RX(0.37), fq.operations.Swap])
def test_lq_native_decomposition_preserves_unitary(gate):
    size = gate.num_subsystems
    source = fq.Program(size)
    source.add(gate, tuple(range(size)))
    target = _SCTarget(
        "test",
        tuple(range(size)),
        frozenset({(0, 1)}) if size == 2 else frozenset(),
        "lqcloud",
    )
    result = lower_sc_to_lq_program(
        normalize_sc_program(snapshot_program(source)), target
    )
    # Remove initial placement to compare semantic operations in logical order.
    inverse = {site: ref.index for ref, site in result.initial_layout}
    lowered = fq.Program(size)
    for op in result.operations:
        lowered.add(op.operation, tuple(inverse[site] for site in op.sites))
        assert op.origin_ids == ("logical.0",)
        assert op.generated_by is None
    sim = fq.simulator.Simulator("unitary", runtime="numpy")
    actual = sim.run(lowered).result().get_unitary()
    expected = sim.run(source).result().get_unitary()
    phase = np.vdot(expected, actual) / len(expected)
    assert np.isclose(abs(phase), 1)
    assert np.allclose(actual, phase * expected)
