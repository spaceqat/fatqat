"""Release checks for the FatQat-to-ZAP neutral-atom pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import fatqat as fq
from fatqat.compiler import compile_qasm_to_na, to_na_simulator_program
from fatqat.compiler.algorithms.zap import load_architecture
from fatqat.compiler.dialects import NAGate, NAMeasure, NAProgram
from fatqat.compiler.dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    TransferEvent,
    ZonedPlan,
    verify_zoned_plan,
)
from fatqat.compiler.passes import schedule_na_with_zap

_SOURCE_ROOT = Path(__file__).parents[2] / "src"


_BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

_GHZ_QASM = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
c = measure q;
"""

# QFT-3 on |101>, with each controlled phase and the final swap decomposed
# into the h/rz/cx subset understood by the current OpenQASM frontend.
_QFT3_QASM = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
x q[0];
x q[2];
h q[0];
rz(pi/4) q[1];
rz(pi/4) q[0];
cx q[1], q[0];
rz(-pi/4) q[0];
cx q[1], q[0];
h q[1];
rz(pi/8) q[2];
rz(pi/8) q[0];
cx q[2], q[0];
rz(-pi/8) q[0];
cx q[2], q[0];
rz(pi/4) q[2];
rz(pi/4) q[1];
cx q[2], q[1];
rz(-pi/4) q[1];
cx q[2], q[1];
h q[2];
cx q[0], q[2];
cx q[2], q[0];
cx q[0], q[2];
c = measure q;
"""


def _default_architecture() -> dict[str, object]:
    return load_architecture("default")


def _scheduled_gate_ids(plan: ZonedPlan) -> list[str]:
    return [
        gate.operation_id
        for event in plan.events
        if type(event) is GateBatch
        for gate in event.gates
    ]


def _measurement_projection(program: NAProgram | ZonedPlan) -> tuple[object, ...]:
    atoms = {atom: index for index, atom in enumerate(program.atoms)}
    clbits = {clbit: index for index, clbit in enumerate(program.clbits)}
    if type(program) is NAProgram:
        measurements = tuple(
            item for item in program.instructions if type(item) is NAMeasure
        )
    else:
        measurements = program.terminal_measurements
    return tuple(
        (
            item.operation_id,
            item.origin_ids,
            atoms[item.atom],
            clbits[item.clbit],
        )
        for item in measurements
    )


def _normalized_reference_program(source: NAProgram) -> fq.Program:
    registers: list[fq.QuantumRegister] = []
    for atom in source.atoms:
        if all(atom.register is not register for register in registers):
            registers.append(atom.register)
    program = fq.Program(registers)
    for instruction in source.instructions:
        if type(instruction) is NAGate:
            program.add(instruction.operation, instruction.atoms)
    return program


def _statevector(
    program: fq.Program,
    *,
    atom_array: bool,
    resource_layout=None,
) -> np.ndarray:
    if atom_array:
        backend = fq.simulator.AtomArraySimulator(runtime="numpy")
    else:
        backend = fq.simulator.Simulator("SV", runtime="numpy")
    job = backend.run(
        program,
        resource_layout=resource_layout,
        result_config={"counts": False, "final_state": True},
    )
    return job.result().get_statevector()


def _assert_same_state(actual: np.ndarray, expected: np.ndarray) -> None:
    pivot = int(np.argmax(np.abs(expected)))
    phase = actual[pivot] / expected[pivot]
    assert np.allclose(actual, phase / abs(phase) * expected, atol=1e-10)


@pytest.mark.parametrize(
    "qasm",
    (_BELL_QASM, _GHZ_QASM, _QFT3_QASM),
    ids=("bell", "ghz-3", "decomposed-qft-3"),
)
def test_real_zap_pipeline_preserves_provenance_measurements_and_semantics(qasm):
    architecture = _default_architecture()
    normalized = compile_qasm_to_na(
        qasm,
        architecture,
        emit=NAProgram.IR_ID,
    ).output
    plan = compile_qasm_to_na(qasm, architecture).output

    expected_ids = [
        item.operation_id for item in normalized.instructions if type(item) is NAGate
    ]
    assert Counter(_scheduled_gate_ids(plan)) == Counter(expected_ids)
    assert all(count == 1 for count in Counter(_scheduled_gate_ids(plan)).values())
    verify_zoned_plan(plan)
    assert _measurement_projection(plan) == _measurement_projection(normalized)

    projected, layout = to_na_simulator_program(replace(plan, terminal_measurements=()))
    assert tuple(layout.device_label(atom) for atom in plan.atoms) == tuple(
        range(len(plan.atoms))
    )
    _assert_same_state(
        _statevector(projected, atom_array=True, resource_layout=layout),
        _statevector(_normalized_reference_program(normalized), atom_array=False),
    )


def _canonical_plan(plan: ZonedPlan) -> tuple[object, ...]:
    atom_index = {atom: index for index, atom in enumerate(plan.atoms)}
    clbit_index = {clbit: index for index, clbit in enumerate(plan.clbits)}
    events: list[object] = []
    for event in plan.events:
        if type(event) is TransferEvent:
            events.append(
                (
                    "transfer",
                    event.kind,
                    tuple(atom_index[atom] for atom in event.atoms),
                    event.positions,
                    event.durations,
                )
            )
        elif type(event) is MoveEvent:
            events.append(
                (
                    "move",
                    event.kind,
                    tuple(atom_index[atom] for atom in event.atoms),
                    event.starts,
                    event.ends,
                    event.distances,
                    event.durations,
                )
            )
        elif type(event) is GateBatch:
            events.append(
                (
                    "gate_batch",
                    event.stage,
                    event.duration,
                    tuple(
                        (
                            gate.operation_id,
                            gate.origin_ids,
                            type(gate.operation).__name__,
                            getattr(gate.operation, "theta", None),
                            tuple(atom_index[atom] for atom in gate.atoms),
                            gate.positions,
                        )
                        for gate in event.gates
                    ),
                )
            )
        elif type(event) is CrosstalkEvent:
            events.append(
                (
                    "crosstalk",
                    tuple(atom_index[atom] for atom in event.atoms),
                    event.positions,
                    event.durations,
                )
            )
        else:  # pragma: no cover - the public verifier owns unknown event types
            raise AssertionError(f"unexpected event type: {type(event).__name__}")
    return (
        tuple(
            (atom_index[atom], position) for atom, position in plan.initial_placement
        ),
        tuple(events),
        tuple(
            (
                item.operation_id,
                item.origin_ids,
                atom_index[item.atom],
                clbit_index[item.clbit],
            )
            for item in plan.terminal_measurements
        ),
    )


def test_real_zap_plan_is_deterministic_in_one_process():
    architecture = _default_architecture()
    normalized = compile_qasm_to_na(
        _GHZ_QASM,
        architecture,
        emit=NAProgram.IR_ID,
    ).output

    first = schedule_na_with_zap(normalized, architecture)
    second = schedule_na_with_zap(normalized, architecture)

    assert first == second
    assert _canonical_plan(first) == _canonical_plan(second)


def test_real_zap_plan_is_deterministic_across_python_hash_seeds():
    outputs = [
        subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--emit-plan"],
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": str(_SOURCE_ROOT),
            },
            text=True,
        )
        for seed in ("1", "73")
    ]

    assert outputs[0] == outputs[1]


def _emit_plan() -> None:
    plan = compile_qasm_to_na(_GHZ_QASM, _default_architecture()).output
    print(json.dumps(_canonical_plan(plan), separators=(",", ":")))


if __name__ == "__main__" and sys.argv[1:] == ["--emit-plan"]:
    _emit_plan()
