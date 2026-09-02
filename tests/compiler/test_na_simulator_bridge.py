"""Tests for projecting neutral-atom physical plans into simulator programs."""

import math

import pytest

import fatqat as fq
from fatqat.compiler import ValidationError, to_na_simulator_program
from fatqat.compiler.dialects import NAMeasure
from fatqat.compiler.dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    ScheduledGate,
    TransferEvent,
    ZonedPlan,
)
from fatqat.operations import PutGate
from fatqat.simulator import AtomArraySimulator


def test_bridge_loads_atoms_once_then_pairs_each_cz_batch():
    plan, atoms, bits = _two_independent_cz_plan()

    program, layout = to_na_simulator_program(plan)

    assert [
        step.operation.name if hasattr(step, "operation") else type(step).__name__
        for step in program._instructions
    ] == [
        "Put",
        "Pair",
        "Pair",
        "CZ",
        "CZ",
        "Unpair",
        "Unpair",
        "Measurement",
        "Measurement",
    ]
    put_steps = [
        step
        for step in program._instructions
        if hasattr(step, "operation") and isinstance(step.operation, PutGate)
    ]
    assert len(put_steps) == 1
    assert put_steps[0].targets == plan.atoms
    assert not any(
        isinstance(step.operation, PutGate)
        for step in program._instructions[1:]
        if hasattr(step, "operation")
    )
    assert program.quantum_registers == (atoms[0].register,)
    assert program.classical_registers == (bits[0].register,)
    assert tuple(layout.device_label(atom) for atom in plan.atoms) == (0, 1, 2, 3)


def test_bridge_preserves_original_registers_in_first_declaration_order():
    first = fq.QuantumRegister(2, name="first")
    second = fq.QuantumRegister(2, name="second")
    bits_first = fq.ClassicalRegister(1, name="bits_first")
    bits_second = fq.ClassicalRegister(1, name="bits_second")
    atoms = (second[1], first[0], second[0], first[1])
    clbits = (bits_second[0], bits_first[0])
    positions = tuple((float(index), 0.0) for index in range(4))
    plan = ZonedPlan(
        atoms=atoms,
        clbits=clbits,
        initial_placement=tuple(zip(atoms, positions, strict=True)),
        events=(),
        terminal_measurements=(
            NAMeasure("na.0", ("logical.0",), atoms[0], clbits[0]),
            NAMeasure("na.1", ("logical.1",), atoms[1], clbits[1]),
        ),
    )

    program, layout = to_na_simulator_program(plan)

    assert program.quantum_registers == (second, first)
    assert program.classical_registers == (bits_second, bits_first)
    assert program._instructions[-2].targets[0] is atoms[0]
    assert program._instructions[-1].targets[0] is atoms[1]
    assert tuple(layout.device_label(atom) for atom in atoms) == (0, 1, 2, 3)


def test_bridge_verifies_the_zoned_plan_boundary():
    atom = fq.QuantumRegister(1, name="atom")[0]
    plan = ZonedPlan(
        atoms=(atom,),
        clbits=(),
        initial_placement=(),
        events=(),
        terminal_measurements=(),
    )

    with pytest.raises(ValidationError, match="initial placement"):
        to_na_simulator_program(plan)


def test_bridge_rejects_a_partial_quantum_register_declaration():
    atoms = fq.QuantumRegister(2, name="atoms")
    plan = ZonedPlan(
        atoms=(atoms[0],),
        clbits=(),
        initial_placement=((atoms[0], (0.0, 0.0)),),
        events=(),
        terminal_measurements=(),
    )

    with pytest.raises(ValidationError, match="each referenced quantum register"):
        to_na_simulator_program(plan)


def test_bridge_rejects_a_partial_classical_register_declaration():
    atom = fq.QuantumRegister(1, name="atom")[0]
    bits = fq.ClassicalRegister(2, name="bits")
    plan = ZonedPlan(
        atoms=(atom,),
        clbits=(bits[0],),
        initial_placement=((atom, (0.0, 0.0)),),
        events=(),
        terminal_measurements=(NAMeasure("na.0", ("logical.0",), atom, bits[0]),),
    )

    with pytest.raises(ValidationError, match="each referenced classical register"):
        to_na_simulator_program(plan)


def test_bridge_checks_each_register_once_when_validating_declarations(monkeypatch):
    atoms = fq.QuantumRegister(3, name="atoms")
    bits = fq.ClassicalRegister(3, name="bits")
    atom_refs = tuple(atoms[index] for index in range(3))
    clbit_refs = tuple(bits[index] for index in range(3))
    plan = ZonedPlan(
        atoms=atom_refs,
        clbits=clbit_refs,
        initial_placement=tuple(
            (atom, (float(index), 0.0)) for index, atom in enumerate(atom_refs)
        ),
        events=(),
        terminal_measurements=tuple(
            NAMeasure(f"na.{index}", (f"logical.{index}",), atom, clbit)
            for index, (atom, clbit) in enumerate(
                zip(atom_refs, clbit_refs, strict=True)
            )
        ),
    )
    calls = {"quantum": 0, "classical": 0}
    quantum_getitem = fq.QuantumRegister.__getitem__
    classical_getitem = fq.ClassicalRegister.__getitem__

    def count_quantum_getitem(register, index):
        if register is atoms:
            calls["quantum"] += 1
        return quantum_getitem(register, index)

    def count_classical_getitem(register, index):
        if register is bits:
            calls["classical"] += 1
        return classical_getitem(register, index)

    monkeypatch.setattr(fq.QuantumRegister, "__getitem__", count_quantum_getitem)
    monkeypatch.setattr(fq.ClassicalRegister, "__getitem__", count_classical_getitem)

    program, layout = to_na_simulator_program(plan)

    assert calls == {"quantum": 3, "classical": 3}
    assert program.quantum_registers == (atoms,)
    assert program.classical_registers == (bits,)
    assert tuple(layout.device_label(atom) for atom in atom_refs) == (0, 1, 2)


def test_bridge_runs_bell_state_with_initial_atom_load():
    plan = _bell_plan()

    program, layout = to_na_simulator_program(plan)
    backend = AtomArraySimulator()
    _lowered, _facts, initial_occupied = backend._prepare_program(program)
    counts = (
        backend.run(
            program,
            shots=1_000,
            resource_layout=layout,
            simulation_config={"seed": 5, "shot_parallelism": "serial"},
        )
        .result()
        .get_counts()
    )

    assert set(counts) <= {"00", "11"}
    assert counts
    assert initial_occupied == frozenset()


def _two_independent_cz_plan():
    atoms = fq.QuantumRegister(4, name="atoms")
    bits = fq.ClassicalRegister(2, name="bits")
    refs = tuple(atoms[index] for index in range(4))
    positions = tuple((float(10 * index), 0.0) for index in range(4))
    moved = tuple((position[0], 5.0) for position in positions)
    return (
        ZonedPlan(
            atoms=refs,
            clbits=(bits[0], bits[1]),
            initial_placement=tuple(zip(refs, positions, strict=True)),
            events=(
                TransferEvent("activate", refs, positions, (0.1, 0.1, 0.1, 0.1)),
                MoveEvent(
                    "big_move",
                    refs,
                    positions,
                    moved,
                    (5.0, 5.0, 5.0, 5.0),
                    (1.0, 1.0, 1.0, 1.0),
                ),
                GateBatch(
                    0,
                    (
                        ScheduledGate(
                            "na.0",
                            ("logical.0",),
                            fq.operations.CZ,
                            (refs[0], refs[1]),
                            moved[:2],
                        ),
                        ScheduledGate(
                            "na.1",
                            ("logical.1",),
                            fq.operations.CZ,
                            (refs[2], refs[3]),
                            moved[2:],
                        ),
                    ),
                    1.0,
                ),
                CrosstalkEvent((refs[0],), (moved[0],), (0.2,)),
                TransferEvent("deactivate", refs, moved, (0.1, 0.1, 0.1, 0.1)),
            ),
            terminal_measurements=(
                NAMeasure("na.2", ("logical.2",), refs[0], bits[0]),
                NAMeasure("na.3", ("logical.3",), refs[2], bits[1]),
            ),
        ),
        refs,
        (bits[0], bits[1]),
    )


def _bell_plan():
    atoms = fq.QuantumRegister(2, name="atoms")
    bits = fq.ClassicalRegister(2, name="bits")
    atom0, atom1 = atoms[0], atoms[1]
    positions = ((0.0, 0.0), (6.0, 0.0))

    def one_qubit_gate(operation_id, operation, atom):
        return GateBatch(
            int(operation_id.removeprefix("na.")),
            (
                ScheduledGate(
                    operation_id,
                    (f"logical.{operation_id}",),
                    operation,
                    (atom,),
                    (positions[atom.index],),
                ),
            ),
            1.0,
        )

    return ZonedPlan(
        atoms=(atom0, atom1),
        clbits=(bits[0], bits[1]),
        initial_placement=((atom0, positions[0]), (atom1, positions[1])),
        events=(
            one_qubit_gate("na.0", fq.operations.RZ(math.pi), atom0),
            one_qubit_gate("na.1", fq.operations.RY(math.pi / 2), atom0),
            one_qubit_gate("na.2", fq.operations.RZ(math.pi), atom1),
            one_qubit_gate("na.3", fq.operations.RY(math.pi / 2), atom1),
            GateBatch(
                4,
                (
                    ScheduledGate(
                        "na.4",
                        ("logical.na.4",),
                        fq.operations.CZ,
                        (atom0, atom1),
                        positions,
                    ),
                ),
                1.0,
            ),
            one_qubit_gate("na.5", fq.operations.RZ(math.pi), atom1),
            one_qubit_gate("na.6", fq.operations.RY(math.pi / 2), atom1),
        ),
        terminal_measurements=(
            NAMeasure("na.7", ("logical.7",), atom0, bits[0]),
            NAMeasure("na.8", ("logical.8",), atom1, bits[1]),
        ),
    )
