"""Compile-only scalability smoke tests for the real ZAP integration."""

from __future__ import annotations

from collections import Counter

import pytest

from fatqat.compiler import compile_qasm_to_na
from fatqat.compiler.algorithms.zap import load_architecture
from fatqat.compiler.dialects import NAGate, NAProgram
from fatqat.compiler.dialects.na_zoned import GateBatch, verify_zoned_plan


def _default_architecture() -> dict[str, object]:
    return load_architecture("default")


def _interaction_qasm(atom_count: int) -> str:
    interactions = "\n".join(
        f"cz q[{first}], q[{first + 1}];" for first in range(0, atom_count, 2)
    )
    return f"""
OPENQASM 3.0;
qubit[{atom_count}] q;
{interactions}
"""


@pytest.mark.parametrize("atom_count", (20, 50))
def test_internal_scheduler_covers_every_large_circuit_gate_once(atom_count):
    architecture = _default_architecture()
    qasm = _interaction_qasm(atom_count)
    normalized = compile_qasm_to_na(
        qasm,
        architecture,
        emit=NAProgram.IR_ID,
    ).output
    plan = compile_qasm_to_na(qasm, architecture).output

    expected_ids = [
        item.operation_id for item in normalized.instructions if type(item) is NAGate
    ]
    scheduled_ids = [
        gate.operation_id
        for event in plan.events
        if type(event) is GateBatch
        for gate in event.gates
    ]
    assert len(plan.atoms) == atom_count
    assert Counter(scheduled_ids) == Counter(expected_ids)
    assert all(count == 1 for count in Counter(scheduled_ids).values())
    verify_zoned_plan(plan)
