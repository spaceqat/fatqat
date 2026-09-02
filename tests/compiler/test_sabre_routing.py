import pytest

import fatqat as fq
from fatqat.compiler.algorithms import ExecuteNode, RouteSwap, sabre_map
from fatqat.compiler.dialects import MeasureOp
from fatqat.compiler.passes import normalize_sc_program, snapshot_program

LINE_EDGES = frozenset(((0, 1), (1, 0), (1, 2), (2, 1)))


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


def test_sabre_executes_adjacent_gate_without_route_swap():
    program = _sc_program(2, ((fq.operations.CZ, (0, 1)),))

    result = sabre_map(
        program,
        sites=(0, 1),
        couplings=frozenset(((0, 1), (1, 0))),
        seed=7,
    )

    assert len(result.events) == 1
    assert isinstance(result.events[0], ExecuteNode)
    assert result.events[0].node_id == 0
    assert set(result.events[0].sites) == {0, 1}


def test_sabre_routes_non_embeddable_triangle_on_line():
    program = _triangle_program()

    result = sabre_map(program, sites=(0, 1, 2), couplings=LINE_EDGES, seed=7)

    assert any(isinstance(event, RouteSwap) for event in result.events)
    assert len(
        [event for event in result.events if isinstance(event, ExecuteNode)]
    ) == len(program.nodes)
    legal_edges = {frozenset(edge) for edge in LINE_EDGES}
    for event in result.events:
        if isinstance(event, RouteSwap) or len(event.sites) == 2:
            assert frozenset(event.sites) in legal_edges


def test_sabre_measurement_uses_site_at_execution_time():
    program = _triangle_program(measure=True)

    result = sabre_map(program, sites=(0, 1, 2), couplings=LINE_EDGES, seed=5)

    final_layout = dict(result.final_layout)
    for event in result.events:
        if not isinstance(event, ExecuteNode):
            continue
        node = program.nodes[event.node_id]
        if type(node.instruction) is MeasureOp:
            assert event.sites == (final_layout[node.qubits[0]],)


def test_sabre_is_reproducible_for_fixed_seed():
    program = _triangle_program(measure=True)

    first = sabre_map(program, sites=(0, 1, 2), couplings=LINE_EDGES, seed=11)
    second = sabre_map(program, sites=(0, 1, 2), couplings=LINE_EDGES, seed=11)

    assert first == second


def test_sabre_rejects_too_few_sites():
    program = _triangle_program()

    with pytest.raises(ValueError, match="sites"):
        sabre_map(program, sites=(0, 1), couplings=frozenset(((0, 1),)), seed=0)


def test_sabre_rejects_interaction_unreachable_in_coupling_graph():
    program = _triangle_program()

    with pytest.raises(ValueError, match="unreachable"):
        sabre_map(
            program,
            sites=(0, 1, 2),
            couplings=frozenset(((0, 1), (1, 0))),
            seed=0,
        )
