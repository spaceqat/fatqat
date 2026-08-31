"""Tests for the fatqat Qiskit BackendV2 integration.

Run with visible print output:

    pytest tests/test_qiskit_integration.py -s -v
"""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

# pylint: disable=wrong-import-position  # imports require the guard above

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate
from qiskit.primitives import BackendSamplerV2

import fatqat as fq
import fatqat.operations as ops
from fatqat.program import _AppliedOperation
from fatqat.qiskit import (
    FatqatBackend,
    QiskitBackendError,
    QiskitConversionError,
    circuit_to_program,
)


def _print_program(program: fq.Program, *, title: str) -> None:
    """Print a short summary of a converted fatqat program."""
    print(f"\n=== {title} ===")
    print(f"quantum registers: {[r.size for r in program.quantum_registers]}")
    print(f"classical registers: {[r.size for r in program.classical_registers]}")
    for index, step in enumerate(program._instructions):
        if isinstance(step, fq.Measurement):
            print(f"  [{index}] measure q={step.targets} -> c={step.outputs}")
        elif isinstance(step, _AppliedOperation):
            print(f"  [{index}] {step.operation.name} on {step.targets}")
        else:
            print(f"  [{index}] {step!r}")


def test_converter_bell_circuit():
    circuit = QuantumCircuit(2, 2, name="bell")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    program = circuit_to_program(circuit)
    _print_program(program, title="Bell circuit conversion")

    assert len(program.quantum_registers) == 1
    assert program.quantum_registers[0].size == 2
    assert len(program._instructions) == 4
    assert program._instructions[0].operation.name == "H"
    assert program._instructions[1].operation.name == "CX"
    assert isinstance(program._instructions[2], fq.Measurement)
    assert isinstance(program._instructions[3], fq.Measurement)


def test_converter_u_gate_maps_to_fatqat_u():
    circuit = QuantumCircuit(1, name="u_gate")
    circuit.append(UGate(0.2, 0.3, 0.4), [0])

    program = circuit_to_program(circuit)
    step = program._instructions[0]
    print("\n=== U gate conversion ===")
    print(f"fatqat operation: {step.operation!r}")

    assert isinstance(step, _AppliedOperation)
    assert isinstance(step.operation, ops.U)
    assert step.operation.theta == pytest.approx(0.2)
    assert step.operation.phi == pytest.approx(0.3)
    assert step.operation.lam == pytest.approx(0.4)


def test_converter_rejects_unbound_parameter():
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rx(theta, 0)

    print("\n=== unbound parameter rejection ===")
    with pytest.raises(QiskitConversionError, match="unbound parameter") as exc:
        circuit_to_program(circuit)
    print(f"expected error: {exc.value}")


def test_converter_wraps_unbound_global_phase():
    phase = Parameter("phase")
    circuit = QuantumCircuit(1, name="unbound_global_phase")
    circuit.global_phase = phase

    with pytest.raises(
        QiskitConversionError,
        match=r"circuit 'unbound_global_phase': global phase has unbound "
        r"parameter\(s\): phase",
    ) as exc_info:
        circuit_to_program(circuit)

    assert exc_info.value.__cause__ is not None


def test_converter_barrier_is_noop():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.barrier()
    circuit.x(0)

    program = circuit_to_program(circuit)
    gate_names = [
        step.operation.name
        for step in program._instructions
        if isinstance(step, _AppliedOperation)
    ]
    print("\n=== barrier no-op conversion ===")
    print(f"gate sequence: {gate_names}")

    assert gate_names == ["H", "X"]


def test_converter_reset():
    circuit = QuantumCircuit(1)
    circuit.reset(0)

    program = circuit_to_program(circuit)
    step = program._instructions[0]
    print("\n=== reset conversion ===")
    print(f"operation: {step.operation.name}")

    assert isinstance(step, _AppliedOperation)
    assert step.operation.name == "Reset"


def test_backend_run_counts():
    circuit = QuantumCircuit(2, 2, name="bell_counts")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    backend = FatqatBackend(method="statevector")
    pass_manager = generate_preset_pass_manager(backend=backend)
    isa_circuit = pass_manager.run(circuit)

    result = backend.run(isa_circuit, shots=100, seed_simulator=7).result()
    counts = result.get_counts()
    print("\n=== backend.run counts ===")
    print("shots: 100, seed: 7")
    print(f"counts: {counts}")

    assert counts
    total = sum(counts.values())
    assert total == 100


def test_backend_multiple_classical_registers_counts_format():
    from qiskit import ClassicalRegister, QuantumRegister

    q = QuantumRegister(2, "q")
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    circuit = QuantumCircuit(q, a, b)
    circuit.x(q[0])
    circuit.measure(q[0], a[0])
    circuit.measure(q[1], b[0])

    backend = FatqatBackend(method="statevector")
    result = backend.run(circuit, shots=1024, seed_simulator=0).result()
    counts = result.get_counts()
    print("\n=== multi-register counts ===")
    print(f"counts: {counts}")

    assert counts
    assert all(" " in key for key in counts)
    assert sum(counts.values()) == 1024


def test_backend_run_no_classical_bits_omits_counts_and_memory():
    circuit = QuantumCircuit(2)
    backend = FatqatBackend(method="statevector")
    result = backend.run(circuit, shots=3, memory=True).result()
    experiment = result.results[0]
    print("\n=== no classical bits ===")
    print(f"data: {experiment.data.to_dict()}")
    print(f"memory_slots: {experiment.header['memory_slots']}")

    assert experiment.header["memory_slots"] == 0
    assert experiment.data.to_dict() == {}


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("shots", 1.5),
        ("shots", True),
        ("memory", "false"),
        ("seed_simulator", 2.8),
    ],
)
def test_backend_rejects_invalid_run_options(option, value):
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    backend = FatqatBackend(method="statevector")
    kwargs = {option: value}
    print(f"\n=== invalid run option {option}={value!r} ===")
    with pytest.raises(QiskitBackendError):
        backend.run(circuit, **kwargs)


def test_backend_rejects_negative_seed_before_returning_job():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    backend = FatqatBackend(method="statevector")

    with pytest.raises(
        QiskitBackendError,
        match=r"seed_simulator must be >= 0, got -1",
    ):
        backend.run(circuit, seed_simulator=-1)


def test_converter_rejects_standalone_qubit():
    from qiskit.circuit import Qubit

    qubit = Qubit()
    circuit = QuantumCircuit([qubit])
    circuit.x(qubit)

    print("\n=== standalone qubit rejection ===")
    with pytest.raises(QiskitConversionError, match="standalone qubit") as exc:
        circuit_to_program(circuit)
    print(f"expected error: {exc.value}")


def test_backend_run_memory_entries():
    circuit = QuantumCircuit(2, 2, name="bell_memory")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    backend = FatqatBackend(method="statevector")
    result = backend.run(circuit, shots=8, memory=True, seed_simulator=3).result()
    memory = result.results[0].data.memory
    counts = result.get_counts()
    print("\n=== backend.run memory ===")
    print("shots: 8, seed: 3")
    print(f"counts: {counts}")
    print(f"memory ({len(memory)} entries): {memory}")

    assert len(memory) == 8


def test_backend_sampler_v2():
    circuit = QuantumCircuit(2, 2, name="bell_sampler")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    backend = FatqatBackend(method="statevector")
    pass_manager = generate_preset_pass_manager(backend=backend)
    isa_circuit = pass_manager.run(circuit)

    sampler = BackendSamplerV2(backend=backend)
    sampler_result = sampler.run([isa_circuit], shots=64).result()
    print("\n=== BackendSamplerV2 ===")
    print("shots: 64")
    for pub_result in sampler_result:
        print(f"  data keys: {list(pub_result.data.keys())}")
        for key, value in pub_result.data.items():
            print(f"  {key}: shape={getattr(value, 'shape', None)}, sample={value}")

    assert sampler_result is not None


def test_u_matrix_matches_qiskit_statevector():
    from qiskit.quantum_info import Operator

    theta, phi, lam = 0.5, 0.25, -0.75
    program = fq.Program(1)
    program.add(ops.U(theta, phi, lam), 0)

    fatqat_state = (
        fq.simulator.Simulator("SV")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    qiskit_matrix = Operator(UGate(theta, phi, lam)).data
    qiskit_state = qiskit_matrix[:, 0]
    print("\n=== U gate statevector parity ===")
    print(f"theta={theta}, phi={phi}, lam={lam}")
    print(f"fatqat statevector: {fatqat_state}")
    print(f"qiskit statevector: {qiskit_state}")

    assert fatqat_state == pytest.approx(qiskit_state)


def _attach_legacy_condition(circuit, index, condition):
    """Re-create a Qiskit 1.x ``c_if`` instruction on a Qiskit 2.x circuit."""
    instruction = circuit.data[index]
    operation = instruction.operation.to_mutable()
    operation.condition = condition
    circuit.data[index] = instruction.replace(operation=operation)


def test_converter_legacy_c_if_whole_register():
    circuit = QuantumCircuit(2, 2, name="legacy_cond")
    circuit.x(0)
    _attach_legacy_condition(circuit, 0, (circuit.cregs[0], 2))

    program = circuit_to_program(circuit)
    step = program._instructions[0]
    creg = program.classical_registers[0]
    print("\n=== legacy c_if whole-register conversion ===")
    print(f"condition: {step.condition}")

    assert step.operation.name == "X"
    assert step.condition == ((creg[0], 0), (creg[1], 1))


def test_converter_legacy_c_if_single_clbit():
    circuit = QuantumCircuit(1, 1, name="legacy_cond_bit")
    circuit.x(0)
    _attach_legacy_condition(circuit, 0, (circuit.clbits[0], 1))

    program = circuit_to_program(circuit)
    step = program._instructions[0]
    creg = program.classical_registers[0]

    assert step.condition == ((creg[0], 1),)


def test_converter_legacy_c_if_value_overflow_rejected():
    circuit = QuantumCircuit(1, 1, name="legacy_cond_overflow")
    circuit.x(0)
    _attach_legacy_condition(circuit, 0, (circuit.cregs[0], 2))

    with pytest.raises(QiskitConversionError, match="does not fit"):
        circuit_to_program(circuit)


def test_converter_legacy_c_if_on_measure_rejected():
    circuit = QuantumCircuit(1, 1, name="legacy_cond_measure")
    circuit.measure(0, 0)
    _attach_legacy_condition(circuit, 0, (circuit.cregs[0], 1))

    with pytest.raises(QiskitConversionError, match="conditional measurement"):
        circuit_to_program(circuit)


def test_backend_batch_circuits_get_distinct_seeds():
    circuit = QuantumCircuit(1, 1, name="coin")
    circuit.h(0)
    circuit.measure(0, 0)

    backend = FatqatBackend()
    result = backend.run(
        [circuit, circuit.copy(), circuit.copy()], shots=200, seed_simulator=7
    ).result()
    counts = [result.get_counts(i) for i in range(3)]
    print("\n=== batch seed independence ===")
    print(f"counts per experiment: {counts}")

    assert len({tuple(sorted(c.items())) for c in counts}) > 1

    repeat = backend.run(
        [circuit, circuit.copy(), circuit.copy()], shots=200, seed_simulator=7
    ).result()
    assert [repeat.get_counts(i) for i in range(3)] == counts


def test_memory_entries_without_counts_raises_instead_of_fabricating():
    from fatqat.qiskit.result import _memory_entries

    program = fq.Program(1)
    program.add(ops.H, 0)
    fatqat_result = (
        fq.simulator.Simulator("SV")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
    )

    with pytest.raises(QiskitBackendError, match="memory=True requires counts"):
        _memory_entries(fatqat_result, shots=10)


def test_adapter_errors_are_also_qiskit_errors():
    from qiskit.exceptions import QiskitError

    from fatqat.errors import FatqatError

    assert issubclass(QiskitBackendError, QiskitError)
    assert issubclass(QiskitBackendError, FatqatError)
    assert issubclass(QiskitConversionError, QiskitError)
    assert issubclass(QiskitConversionError, FatqatError)

    backend = FatqatBackend()
    with pytest.raises(QiskitError):
        backend.run(QuantumCircuit(1, 1), shots=-1)


def test_provider_backends_report_their_provider():
    from fatqat.qiskit import FatqatProvider

    provider = FatqatProvider()
    backend = provider.get_backend("fatqat_simulator")
    assert backend.provider is provider


def test_result_header_labels_use_standard_pairs():
    circuit = QuantumCircuit(2, 2, name="labels")
    circuit.h(0)
    circuit.measure([0, 1], [0, 1])

    result = FatqatBackend().run(circuit, shots=10).result()
    header = result.results[0].header
    header = header if isinstance(header, dict) else header.__dict__
    assert header["qubit_labels"] == [["q", 0], ["q", 1]]
    assert header["clbit_labels"] == [["c", 0], ["c", 1]]


def test_batch_result_metadata_reports_each_circuit_seed():
    circuit = QuantumCircuit(1, 1, name="seed_meta")
    circuit.h(0)
    circuit.measure(0, 0)

    result = FatqatBackend().run(
        [circuit, circuit.copy(), circuit.copy()], shots=50, seed_simulator=7
    ).result()
    reported = [experiment.seed_simulator for experiment in result.results]
    assert reported == [7, 8, 9]

    # a reported seed must actually reproduce its experiment
    replay = FatqatBackend().run(circuit, shots=50, seed_simulator=9).result()
    assert replay.get_counts() == result.get_counts(2)
