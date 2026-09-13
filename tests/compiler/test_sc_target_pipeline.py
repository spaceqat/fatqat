import warnings
from typing import get_type_hints

import pytest

import fatqat as fq
from fatqat.compiler import (
    compile_qasm_to_sc,
    to_sc_simulator_program,
)
from fatqat.compiler.dialects import NativeMeasure, SCNativeProgram
from fatqat.compiler.dialects.sc_native import _RotationNativeProgram
from fatqat.compiler.pipelines import _compile_qasm_to_sc_rotation
from fatqat.operations import Measurement
from fatqat.simulator import SCQubitSimulator
from fatqat.simulator.fake_superconducting import _SCQubitRotationSimulator

TRIANGLE_QASM = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
x q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[0], q[2];
c = measure q;
"""

LINE = ((0, 1), (1, 2))
FIVE_SITE_LINE = ((0, 1), (1, 2), (2, 3), (3, 4))


@pytest.mark.parametrize(
    ("backend", "compile_qasm", "program_type", "pass_name"),
    (
        (
            SCQubitSimulator(num_qubits=3, couplings=LINE, runtime="numpy"),
            compile_qasm_to_sc,
            SCNativeProgram,
            "lower-sc-to-native",
        ),
        (
            _SCQubitRotationSimulator(num_qubits=3, couplings=LINE, runtime="numpy"),
            _compile_qasm_to_sc_rotation,
            _RotationNativeProgram,
            "lower-sc-to-rotation",
        ),
    ),
)
@pytest.mark.parametrize("split_registers", (False, True))
def test_qasm_target_pipeline_preserves_public_order_through_routing(
    backend, compile_qasm, program_type, pass_name, split_registers
):
    source = TRIANGLE_QASM
    if split_registers:
        source = source.replace("bit[3] c;", "bit[1] a; bit[2] b;").replace(
            "c = measure q;",
            "b[1] = measure q[2]; b[0] = measure q[1]; a[0] = measure q[0];",
        )
    result = compile_qasm(source, backend, seed=5)

    assert type(result.output) is program_type
    assert result.route == ("parse-qasm", "normalize-sc", pass_name)
    assert [register.name for register in result.program.classical_registers] == (
        ["a", "b"] if split_registers else ["c"]
    )

    counts = (
        backend.run(
            result,
            shots=256,
            simulation_config={"seed": 9},
        )
        .result()
        .get_counts()
    )

    assert counts == {"110": 256}


def test_simulator_bridge_uses_fixed_physical_refs_and_original_clbits():
    backend = SCQubitSimulator(num_qubits=5, couplings=FIVE_SITE_LINE, runtime="numpy")
    native = compile_qasm_to_sc(TRIANGLE_QASM, backend, seed=2).output

    program, resource_layout = to_sc_simulator_program(native)

    used_sites = {site for _qubit, site in native.initial_layout + native.final_layout}
    for instruction in native.operations:
        if hasattr(instruction, "sites"):
            used_sites.update(instruction.sites)
        elif hasattr(instruction, "site"):
            used_sites.add(instruction.site)
    physical_refs = {
        register[index]
        for register in program.quantum_registers
        for index in range(register.size)
    }
    assert resource_layout.refs == physical_refs
    assert resource_layout.device_labels == used_sites

    native_clbits = tuple(
        instruction.clbit
        for instruction in native.operations
        if isinstance(instruction, NativeMeasure)
    )
    simulator_clbits = tuple(
        output
        for instruction in program._instructions
        if isinstance(instruction, Measurement)
        for output in instruction.outputs
    )
    assert simulator_clbits == native_clbits


def test_simulator_bridge_exposes_only_the_canonical_native_type():
    assert get_type_hints(to_sc_simulator_program)["native"] is SCNativeProgram
    with pytest.raises(TypeError, match="native must be SCNativeProgram"):
        to_sc_simulator_program(object())


def _assert_classical_counts(backend, executable, expected, unwritten, **run_options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backend.run(
            executable,
            shots=4,
            simulation_config={"seed": 7, "shot_parallelism": "serial"},
            **run_options,
        ).result()

    assert result.get_counts() == {expected: 4}
    assert result.get_counts_as_tuples() == {tuple(map(int, expected)): 4}
    assert [str(warning.message) for warning in caught] == (
        ["counts contain clbits that were never measured; returning zero-filled counts"]
        if unwritten
        else []
    )
    assert all(warning.category is UserWarning for warning in caught)


@pytest.mark.parametrize(
    ("declarations", "measurements", "expected", "unwritten"),
    (
        ((("a", 1), ("b", 1), ("unused", 1)), ((1, 1, 0), (0, 0, 0)), "100", True),
        ((("a", 1), ("b", 1)), ((1, 1, 0), (0, 0, 0)), "10", False),
        ((("a", 1), ("unused", 1)), ((0, 0, 0),), "10", True),
        ((("a", 2), ("b", 2), ("unused", 1)), ((1, 1, 1), (0, 0, 1)), "01000", True),
        ((("c", 1), ("c", 1)), ((1, 1, 0), (0, 0, 0)), "10", False),
    ),
    ids=("combined", "reversed", "unused", "partially-measured", "same-name"),
)
def test_logical_compilation_preserves_classical_output(
    declarations, measurements, expected, unwritten
):
    qubits = fq.QuantumRegister(2, name="q")
    registers = tuple(
        fq.ClassicalRegister(size, name=name) for name, size in declarations
    )
    logical = fq.LogicalProgram([qubits], registers)
    logical.add(fq.operations.X, qubits[0])
    for qubit, register, index in measurements:
        logical.measure(qubits[qubit], registers[register][index])
    backend = SCQubitSimulator(num_qubits=2, couplings=((0, 1),), runtime="numpy")

    compiled = fq.compiler.compile_to_sc(logical, backend, seed=7)

    assert compiled.program.classical_registers == registers
    _assert_classical_counts(backend, compiled, expected, unwritten)
    bridged, layout = to_sc_simulator_program(compiled.output)
    assert bridged.classical_registers == registers
    _assert_classical_counts(
        backend, bridged, expected, unwritten, resource_layout=layout
    )


@pytest.mark.parametrize(
    ("backend_type", "compile_qasm"),
    (
        (SCQubitSimulator, compile_qasm_to_sc),
        (_SCQubitRotationSimulator, _compile_qasm_to_sc_rotation),
    ),
)
def test_qasm_compilation_and_bridge_preserve_classical_declarations(
    backend_type, compile_qasm
):
    source = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[1] a;
    bit[1] b;
    bit[1] unused;
    x q[0];
    b[0] = measure q[1];
    a[0] = measure q[0];
    """
    backend = backend_type(num_qubits=2, couplings=((0, 1),), runtime="numpy")

    compiled = compile_qasm(source, backend, seed=7)

    assert [register.name for register in compiled.program.classical_registers] == [
        "a",
        "b",
        "unused",
    ]
    _assert_classical_counts(backend, compiled, "100", True)
    bridged, layout = to_sc_simulator_program(compiled.output)
    assert bridged.classical_registers == compiled.program.classical_registers
    _assert_classical_counts(backend, bridged, "100", True, resource_layout=layout)


@pytest.mark.parametrize("has_declarations", (False, True))
def test_compilation_preserves_declarations_without_measurements(has_declarations):
    qubits = fq.QuantumRegister(1, name="q")
    registers = (fq.ClassicalRegister(2, name="unused"),) if has_declarations else ()
    logical = fq.LogicalProgram([qubits], registers)
    logical.add(fq.operations.X, qubits[0])
    backend = SCQubitSimulator(num_qubits=1, couplings=(), runtime="numpy")

    compiled = fq.compiler.compile_to_sc(logical, backend)

    assert compiled.output.classical_registers == registers
    assert compiled.program.classical_registers == registers
    _assert_classical_counts(
        backend,
        compiled,
        "00" if has_declarations else "",
        has_declarations,
        result_config={"counts": True, "final_state": False},
    )


@pytest.mark.parametrize(
    ("backend_type", "compile_qasm"),
    (
        (SCQubitSimulator, compile_qasm_to_sc),
        (_SCQubitRotationSimulator, _compile_qasm_to_sc_rotation),
    ),
)
@pytest.mark.parametrize("has_declarations", (False, True))
def test_qasm_compilation_preserves_declarations_without_measurements(
    backend_type, compile_qasm, has_declarations
):
    declaration = "bit[2] unused;" if has_declarations else ""
    source = f"OPENQASM 3.0; qubit[1] q; {declaration} x q[0];"
    backend = backend_type(num_qubits=1, couplings=(), runtime="numpy")

    compiled = compile_qasm(source, backend, seed=7)

    registers = compiled.program.classical_registers
    assert compiled.output.classical_registers == registers
    assert [(register.name, register.size) for register in registers] == (
        [("unused", 2)] if has_declarations else []
    )
    _assert_classical_counts(
        backend,
        compiled,
        "00" if has_declarations else "",
        has_declarations,
        result_config={"counts": True, "final_state": False},
    )
