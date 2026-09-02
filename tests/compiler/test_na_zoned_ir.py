import pytest

import fatqat as fq

from fatqat.compiler import ValidationError
from fatqat.compiler.dialects import NAGate, NAMeasure
from fatqat.compiler.dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    ScheduledGate,
    ZonedPlan,
    verify_zoned_plan,
)


def test_zoned_plan_replays_movement_before_a_gate_batch():
    atoms = fq.QuantumRegister(2, name="atoms")
    atom0, atom1 = atoms[0], atoms[1]
    plan = ZonedPlan(
        atoms=(atom0, atom1),
        clbits=(),
        initial_placement=((atom0, (0.0, 0.0)), (atom1, (6.0, 0.0))),
        events=(
            MoveEvent(
                "big_move",
                (atom0, atom1),
                ((0.0, 0.0), (6.0, 0.0)),
                ((0.0, 10.0), (4.0, 10.0)),
                (10.0, 10.2),
                (1.0, 1.0),
            ),
            GateBatch(
                0,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.CZ,
                        (atom0, atom1),
                        ((0.0, 10.0), (4.0, 10.0)),
                    ),
                ),
                1.0,
            ),
        ),
        terminal_measurements=(),
    )

    verify_zoned_plan(plan)


def test_zoned_plan_rejects_a_move_with_a_stale_start_position():
    atoms = fq.QuantumRegister(2, name="atoms")
    atom0, atom1 = atoms[0], atoms[1]
    plan = _plan(
        (atom0, atom1),
        (
            MoveEvent(
                "big_move",
                (atom0, atom1),
                ((1.0, 0.0), (6.0, 0.0)),
                ((0.0, 10.0), (4.0, 10.0)),
                (10.0, 10.2),
                (1.0, 1.0),
            ),
        ),
    )

    with pytest.raises(ValidationError, match="move start does not match"):
        verify_zoned_plan(plan)


def test_zoned_plan_rejects_a_gate_at_a_stale_position():
    atoms = fq.QuantumRegister(2, name="atoms")
    atom0, atom1 = atoms[0], atoms[1]
    plan = _plan(
        (atom0, atom1),
        (
            MoveEvent(
                "big_move",
                (atom0, atom1),
                ((0.0, 0.0), (6.0, 0.0)),
                ((0.0, 10.0), (4.0, 10.0)),
                (10.0, 10.2),
                (1.0, 1.0),
            ),
            GateBatch(
                0,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.CZ,
                        (atom0, atom1),
                        ((0.0, 0.0), (6.0, 0.0)),
                    ),
                ),
                1.0,
            ),
        ),
    )

    with pytest.raises(ValidationError, match="scheduled gate position does not match"):
        verify_zoned_plan(plan)


def test_zoned_plan_rejects_an_atom_twice_in_one_gate_batch():
    atoms = fq.QuantumRegister(2, name="atoms")
    atom0, atom1 = atoms[0], atoms[1]
    plan = _plan(
        (atom0, atom1),
        (
            GateBatch(
                0,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.RX(0.5),
                        (atom0,),
                        ((0.0, 0.0),),
                    ),
                    ScheduledGate(
                        "na.1",
                        ("logical.1",),
                        fq.operations.RY(0.5),
                        (atom0,),
                        ((0.0, 0.0),),
                    ),
                ),
                1.0,
            ),
        ),
    )

    with pytest.raises(ValidationError, match="participates twice"):
        verify_zoned_plan(plan)


def test_zoned_plan_rejects_an_undeclared_atom_in_a_gate():
    atoms = fq.QuantumRegister(3, name="atoms")
    atom0, atom1, atom2 = atoms[0], atoms[1], atoms[2]
    plan = _plan(
        (atom0, atom1),
        (
            GateBatch(
                0,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.CZ,
                        (atom0, atom2),
                        ((0.0, 0.0), (10.0, 0.0)),
                    ),
                ),
                1.0,
            ),
        ),
    )

    with pytest.raises(ValidationError, match="undeclared atom"):
        verify_zoned_plan(plan)


def test_zoned_plan_rejects_a_repeated_scheduled_operation_id():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    plan = _plan(
        (atom,),
        (
            GateBatch(
                0,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.RX(0.5),
                        (atom,),
                        ((0.0, 0.0),),
                    ),
                ),
                1.0,
            ),
            GateBatch(
                1,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.1",),
                        fq.operations.RY(0.5),
                        (atom,),
                        ((0.0, 0.0),),
                    ),
                ),
                1.0,
            ),
        ),
    )

    with pytest.raises(ValidationError, match="duplicate scheduled operation ID"):
        verify_zoned_plan(plan)


def test_zoned_plan_rejects_a_gate_in_terminal_measurements():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    plan = ZonedPlan(
        atoms=(atom,),
        clbits=(),
        initial_placement=((atom, (0.0, 0.0)),),
        events=(),
        terminal_measurements=(
            NAGate("na.0", ("logical.0",), fq.operations.RX(0.5), (atom,)),
        ),
    )

    with pytest.raises(ValidationError, match="exact NAMeasure"):
        verify_zoned_plan(plan)


def test_zoned_plan_rejects_a_scheduled_gate_id_used_by_a_measurement():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    clbit = fq.ClassicalRegister(1, name="c")[0]
    plan = ZonedPlan(
        atoms=(atom,),
        clbits=(clbit,),
        initial_placement=((atom, (0.0, 0.0)),),
        events=(
            GateBatch(
                0,
                (
                    ScheduledGate(
                        "na.0",
                        ("logical.0",),
                        fq.operations.RX(0.5),
                        (atom,),
                        ((0.0, 0.0),),
                    ),
                ),
                1.0,
            ),
        ),
        terminal_measurements=(NAMeasure("na.0", ("logical.1",), atom, clbit),),
    )

    with pytest.raises(ValidationError, match="duplicate ZonedPlan operation ID"):
        verify_zoned_plan(plan)


def test_zoned_plan_replays_current_positions_for_crosstalk():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    plan = _plan(
        (atom,),
        (
            MoveEvent(
                "big_move",
                (atom,),
                ((0.0, 0.0),),
                ((4.0, 0.0),),
                (4.0,),
                (1.0,),
            ),
            CrosstalkEvent((atom,), ((4.0, 0.0),), (0.5,)),
        ),
    )

    verify_zoned_plan(plan)


def test_zoned_plan_rejects_crosstalk_at_a_stale_position_after_a_move():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    plan = _plan(
        (atom,),
        (
            MoveEvent(
                "big_move",
                (atom,),
                ((0.0, 0.0),),
                ((4.0, 0.0),),
                (4.0,),
                (1.0,),
            ),
            CrosstalkEvent((atom,), ((0.0, 0.0),), (0.5,)),
        ),
    )

    with pytest.raises(ValidationError, match="crosstalk position does not match"):
        verify_zoned_plan(plan)


def _plan(atoms, events):
    return ZonedPlan(
        atoms=atoms,
        clbits=(),
        initial_placement=tuple(
            (atom, (float(6 * index), 0.0)) for index, atom in enumerate(atoms)
        ),
        events=events,
        terminal_measurements=(),
    )
