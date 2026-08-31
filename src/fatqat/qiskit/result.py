"""Build Qiskit ``Result`` objects from fatqat execution output."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from qiskit.result import Result

from .errors import QiskitBackendError

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from fatqat.result import Result as FatqatResult


def build_qiskit_result(
    *,
    backend_name: str,
    backend_version: str,
    circuits: list[QuantumCircuit],
    fatqat_results: list[FatqatResult],
    shots: int,
    memory: bool,
    seed_simulators: list[int | None],
    success: bool = True,
) -> Result:
    """Package one fatqat result per input circuit into a Qiskit ``Result``.

    ``seed_simulators`` carries the seed each experiment actually ran with
    (they differ per circuit in a batch), so every experiment's metadata is
    individually reproducible.
    """
    job_id = str(uuid.uuid4())
    experiments = []
    for circuit, fatqat_result, circuit_seed in zip(
        circuits, fatqat_results, seed_simulators
    ):
        experiments.append(
            _experiment_result(
                circuit=circuit,
                fatqat_result=fatqat_result,
                shots=shots,
                memory=memory,
                seed_simulator=circuit_seed,
                success=success,
            )
        )
    return Result.from_dict(
        {
            "backend_name": backend_name,
            "backend_version": backend_version,
            "job_id": job_id,
            "success": success,
            "results": experiments,
        }
    )


def _experiment_result(
    *,
    circuit: QuantumCircuit,
    fatqat_result: FatqatResult,
    shots: int,
    memory: bool,
    seed_simulator: int | None,
    success: bool,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if circuit.num_clbits > 0:
        counts_hex = _hex_counts(fatqat_result)
        if counts_hex:
            data["counts"] = counts_hex
        if memory:
            data["memory"] = _memory_entries(fatqat_result, shots)

    header = {
        "name": circuit.name,
        "memory_slots": circuit.num_clbits,
        "n_qubits": circuit.num_qubits,
        "n_clbits": circuit.num_clbits,
        "global_phase": float(circuit.global_phase),
        "qubit_labels": _register_labels(circuit.qregs),
        "clbit_labels": _register_labels(circuit.cregs),
        "creg_sizes": [[creg.name, creg.size] for creg in circuit.cregs],
        "metadata": dict(circuit.metadata) if circuit.metadata else {},
    }
    experiment: dict[str, Any] = {
        "shots": shots,
        "success": success,
        "header": header,
        "data": data,
        "time_taken": 0.0,
        "metadata": dict(fatqat_result.metadata),
        "status": "DONE" if success else "ERROR",
    }
    if seed_simulator is not None:
        experiment["seed_simulator"] = seed_simulator
    return experiment


def _register_labels(registers) -> list[list[Any]]:
    # Qiskit's result-header convention: a flat list of [name, index] pairs,
    # one per bit, not nested per-register label strings.
    return [[reg.name, index] for reg in registers for index in range(reg.size)]


def _hex_counts(fatqat_result: FatqatResult) -> dict[str, int]:
    if "counts" not in fatqat_result.available_data:
        return {}
    counts = fatqat_result.get_counts_as_tuples()
    return {_tuple_to_hex(key): value for key, value in counts.items()}


def _memory_entries(fatqat_result: FatqatResult, shots: int) -> list[str]:
    # fatqat results carry no native per-shot record, so memory is always
    # synthesized from counts; entries are grouped by outcome, not in shot
    # order.
    if "counts" not in fatqat_result.available_data:
        raise QiskitBackendError(
            "memory=True requires counts data, but the fatqat result contains none"
        )
    return _expand_counts_to_memory(fatqat_result.get_counts_as_tuples(), shots)


def _expand_counts_to_memory(
    counts: dict[tuple[int, ...], int],
    shots: int,
) -> list[str]:
    """Compatibility fallback when native per-shot memory is unavailable."""
    memory: list[str] = []
    for key, count in counts.items():
        encoded = _tuple_to_hex(key)
        memory.extend([encoded] * count)
    if len(memory) < shots:
        memory.extend(["0x0"] * (shots - len(memory)))
    return memory[:shots]


def _tuple_to_hex(outcome: tuple[int, ...]) -> str:
    value = sum(int(bit) << index for index, bit in enumerate(outcome))
    return hex(value)
