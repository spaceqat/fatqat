import pytest

import fatqat as fq
from fatqat.compiler import ValidationError
from fatqat.compiler.algorithms.zap import ZapInteraction, ZapTrace
from fatqat.compiler.dialects import NAGate, NAMeasure, NAProgram
from fatqat.compiler.dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    TransferEvent,
)
from fatqat.compiler.passes.na_zap import _project_interactions, _translate_trace


def _init_instruction():
    return {
        "type": "Init",
        "duration": [0.0],
        "locs": [{"id": 0, "x": 0.0, "y": 0.0}],
    }


def _one_qubit_gate_instruction(operation_id, *, position=(0.0, 0.0), stage=0):
    return {
        "type": "1qGate",
        "stage": stage,
        "duration": [1.0],
        "qs": [0],
        "gates": [[0, 0]],
        "operation_ids": [operation_id],
        "locs": [{"id": 0, "x": position[0], "y": position[1]}],
    }


def test_project_interactions_uses_dense_indices_and_preserves_gate_identity():
    atoms = fq.QuantumRegister(3, name="atoms")
    bits = fq.ClassicalRegister(1, name="bits")
    source = NAProgram(
        atoms=(atoms[0], atoms[1], atoms[2]),
        clbits=(bits[0],),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.CZ, (atoms[1], atoms[0])),
            NAMeasure("na.1", ("logical.1",), atoms[2], bits[0]),
        ),
    )

    assert _project_interactions(source) == (ZapInteraction("na.0", (1, 0)),)


def test_translate_trace_restores_atom_refs_and_preserves_all_instruction_families():
    atoms = fq.QuantumRegister(3, name="atoms")
    bits = fq.ClassicalRegister(1, name="bits")
    atom0, atom1, idle_atom = atoms[0], atoms[1], atoms[2]
    source = NAProgram(
        atoms=(atom0, atom1, idle_atom),
        clbits=(bits[0],),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.CZ, (atom1, atom0)),
            NAMeasure("na.1", ("logical.1",), idle_atom, bits[0]),
        ),
    )
    trace = ZapTrace(
        atom_count=3,
        instructions=(
            {
                "type": "Init",
                "duration": [0.0, 0.0, 0.0],
                "locs": [
                    {"id": 0, "x": 0.0, "y": 0.0},
                    {"id": 1, "x": 6.0, "y": 0.0},
                    {"id": 2, "x": 12.0, "y": 0.0},
                ],
            },
            {
                "type": "Activate",
                "qs": [1, 0],
                "duration": [0.2, 0.2],
                "locs": [
                    {"id": 1, "x": 6.0, "y": 0.0},
                    {"id": 0, "x": 0.0, "y": 0.0},
                ],
            },
            {
                "type": "BigMove",
                "qs": [1, 0],
                "distance": [10.0, 10.0],
                "duration": [1.0, 1.0],
                "locs": [
                    {
                        "id": 1,
                        "x_begin": 6.0,
                        "y_begin": 0.0,
                        "x_end": 4.0,
                        "y_end": 10.0,
                    },
                    {
                        "id": 0,
                        "x_begin": 0.0,
                        "y_begin": 0.0,
                        "x_end": 0.0,
                        "y_end": 10.0,
                    },
                ],
            },
            {
                "type": "2qGate",
                "stage": 7,
                "duration": [1.0, 2.0],
                "qs": [1, 0],
                "gates": [[1, 0]],
                "operation_ids": ["na.0"],
                "locs": [
                    {"id": 1, "x": 4.0, "y": 10.0},
                    {"id": 0, "x": 0.0, "y": 10.0},
                ],
            },
            {
                "type": "Crosstalk",
                "qs": [2],
                "duration": [2.0],
                "locs": [{"id": 2, "x": 12.0, "y": 0.0}],
            },
            {
                "type": "Deactivate",
                "qs": [1, 0],
                "duration": [0.2, 0.2],
                "locs": [
                    {"id": 1, "x": 4.0, "y": 10.0},
                    {"id": 0, "x": 0.0, "y": 10.0},
                ],
            },
        ),
    )

    plan = _translate_trace(source, trace)

    assert plan.initial_placement == (
        (atom0, (0.0, 0.0)),
        (atom1, (6.0, 0.0)),
        (idle_atom, (12.0, 0.0)),
    )
    assert [type(event) for event in plan.events] == [
        TransferEvent,
        MoveEvent,
        GateBatch,
        CrosstalkEvent,
        TransferEvent,
    ]
    assert plan.events[0].kind == "activate"
    assert plan.events[0].atoms == (atom1, atom0)
    assert plan.events[0].atoms[0] is atom1
    assert plan.events[0].atoms[1] is atom0
    assert plan.events[1].kind == "big_move"
    assert plan.events[1].atoms == (atom1, atom0)
    assert plan.events[2].gates[0].atoms == (atom1, atom0)
    assert plan.events[2].gates[0].atoms is source.instructions[0].atoms
    assert plan.events[2].gates[0].operation_id == "na.0"
    assert plan.events[2].gates[0].origin_ids == ("logical.0",)
    assert plan.events[2].gates[0].operation is source.instructions[0].operation
    assert plan.events[2].duration == 2.0
    assert plan.events[3].atoms == (idle_atom,)
    assert plan.events[4].kind == "deactivate"
    assert plan.events[4].atoms == (atom1, atom0)
    assert plan.terminal_measurements == (source.instructions[1],)
    assert plan.terminal_measurements[0] is source.instructions[1]


def test_translate_trace_rejects_locations_missing_required_coordinates():
    atom = fq.QuantumRegister(1, name="atom")[0]
    source = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(NAGate("na.0", ("logical.0",), fq.operations.RX(0.5), (atom,)),),
    )
    trace = ZapTrace(
        atom_count=1,
        instructions=(
            {
                "type": "Init",
                "duration": [0.0],
                "locs": [{"id": 0, "y": 0.0}],
            },
            _one_qubit_gate_instruction("na.0"),
        ),
    )

    with pytest.raises(ValidationError, match="missing 'x'"):
        _translate_trace(source, trace)


def test_translate_trace_normalizes_one_qubit_pairs_and_park_events():
    atom = fq.QuantumRegister(1, name="atom")[0]
    source = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(NAGate("na.0", ("logical.0",), fq.operations.RY(0.5), (atom,)),),
    )
    trace = ZapTrace(
        atom_count=1,
        instructions=(
            _init_instruction(),
            {
                "type": "Park",
                "qs": [0],
                "distance": [4.0],
                "duration": [2.0],
                "locs": [
                    {
                        "id": 0,
                        "x_begin": 0.0,
                        "y_begin": 0.0,
                        "x_end": 4.0,
                        "y_end": 0.0,
                    }
                ],
            },
            _one_qubit_gate_instruction("na.0", position=(4.0, 0.0)),
        ),
    )

    plan = _translate_trace(source, trace)

    assert plan.events[0] == MoveEvent(
        "park", (atom,), ((0.0, 0.0),), ((4.0, 0.0),), (4.0,), (2.0,)
    )
    assert plan.events[1].gates[0].atoms == (atom,)


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        (ZapTrace(1, (_init_instruction(),)), "did not execute"),
        (
            ZapTrace(
                1,
                (
                    _init_instruction(),
                    _one_qubit_gate_instruction("not-a-source-id"),
                ),
            ),
            "unknown gate ID",
        ),
        (
            ZapTrace(
                1,
                (
                    _init_instruction(),
                    _one_qubit_gate_instruction("na.0"),
                    _one_qubit_gate_instruction("na.0", stage=1),
                ),
            ),
            "executes gate ID twice",
        ),
        (
            ZapTrace(
                1,
                (
                    _init_instruction(),
                    {
                        "type": "1qGate",
                        "stage": 0,
                        "duration": [],
                        "qs": [],
                        "gates": [],
                        "operation_ids": [],
                        "locs": [],
                    },
                ),
            ),
            "operation_ids must align",
        ),
    ],
)
def test_translate_trace_rejects_incomplete_or_invalid_gate_execution(trace, message):
    atom = fq.QuantumRegister(1, name="atom")[0]
    source = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(NAGate("na.0", ("logical.0",), fq.operations.RX(0.5), (atom,)),),
    )

    with pytest.raises(ValidationError, match=message):
        _translate_trace(source, trace)


def test_translate_trace_rejects_gate_operand_mismatch():
    atoms = fq.QuantumRegister(2, name="atoms")
    source = NAProgram(
        atoms=(atoms[0], atoms[1]),
        clbits=(),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.CZ, (atoms[0], atoms[1])),
        ),
    )
    trace = ZapTrace(
        atom_count=2,
        instructions=(
            {
                "type": "Init",
                "duration": [0.0, 0.0],
                "locs": [
                    {"id": 0, "x": 0.0, "y": 0.0},
                    {"id": 1, "x": 6.0, "y": 0.0},
                ],
            },
            {
                "type": "2qGate",
                "stage": 0,
                "duration": [1.0, 1.0],
                "qs": [1, 0],
                "gates": [[1, 0]],
                "operation_ids": ["na.0"],
                "locs": [
                    {"id": 1, "x": 6.0, "y": 0.0},
                    {"id": 0, "x": 0.0, "y": 0.0},
                ],
            },
        ),
    )

    with pytest.raises(ValidationError, match="operands do not match"):
        _translate_trace(source, trace)
