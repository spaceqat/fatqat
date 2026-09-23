import warnings
from typing import get_type_hints

import pytest

import fatqat as fq
from fatqat.compiler import (
    LOGICAL_SC_PIPELINE,
    SC_PIPELINE,
    CompilationResult,
    CompileContext,
    ExecutableCompilationResult,
    PassError,
    SCTarget,
    compile_qasm_to_sc,
    compile_to_sc,
    create_sc_pipeline,
    to_sc_simulator_program,
)
from fatqat.compiler.dialects import (
    NativeGate,
    NativeMeasure,
    QasmSource,
    SCNativeProgram,
)
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


FULL = ((0, 1), (0, 2), (1, 2))
QASM2_BELL = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    "h q[0]; cx q[0],q[1]; measure q -> c;"
)
SC_ROUTE = ("parse-qasm", "normalize-sc", "lower-sc-to-native")
LOGICAL_ROUTE = ("freeze-logical", "normalize-sc", "lower-sc-to-native")


def _logical_circuit(num_qubits=3):
    logical = fq.LogicalProgram(num_qubits, num_qubits)
    for index in range(num_qubits):
        logical.add(fq.operations.RX(0.17 + 0.13 * index), index)
    if num_qubits == 3:
        for edge in FULL:
            logical.add(fq.operations.CX, edge)
        logical.add(fq.operations.Swap, (0, 2))
    logical.measure_all()
    return logical


def _block_simulator_construction(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("SCQubitSimulator constructed during compilation")

    monkeypatch.setattr(SCQubitSimulator, "__init__", forbidden)


@pytest.mark.parametrize(
    ("compile_with", "route"),
    (
        (lambda target: compile_to_sc(_logical_circuit(), target), LOGICAL_ROUTE),
        (
            lambda target: compile_to_sc(_logical_circuit(), backend=target),
            LOGICAL_ROUTE,
        ),
        (lambda target: compile_qasm_to_sc(TRIANGLE_QASM, target), SC_ROUTE),
        (lambda target: compile_qasm_to_sc(QASM2_BELL, target), SC_ROUTE),
        (
            lambda target: compile_qasm_to_sc(QasmSource(TRIANGLE_QASM), target),
            SC_ROUTE,
        ),
        (
            lambda target: compile_qasm_to_sc(
                TRIANGLE_QASM, backend=target, filename="triangle.qasm"
            ),
            SC_ROUTE,
        ),
    ),
    ids=(
        "logical",
        "logical-backend-keyword",
        "qasm3-text",
        "qasm2-text",
        "qasm-source",
        "qasm-backend-keyword",
    ),
)
def test_convenience_entries_compile_for_a_target_without_a_simulator(
    monkeypatch, compile_with, route
):
    target = SCTarget(num_qubits=3, couplings=LINE)
    _block_simulator_construction(monkeypatch)

    result = compile_with(target)

    assert type(result) is ExecutableCompilationResult
    assert type(result.output) is SCNativeProgram
    assert result.route == route


@pytest.mark.parametrize(
    ("source", "pipeline"),
    (
        (_logical_circuit, LOGICAL_SC_PIPELINE),
        (lambda: QasmSource(TRIANGLE_QASM), SC_PIPELINE),
    ),
    ids=("logical-pipeline", "qasm-pipeline"),
)
def test_explicit_pipeline_compiles_for_a_target_without_packaging(
    monkeypatch, source, pipeline
):
    target = SCTarget(num_qubits=3, couplings=LINE)
    _block_simulator_construction(monkeypatch)

    result = create_sc_pipeline().compile(
        source(), pipeline=pipeline, context=CompileContext(target=target)
    )

    # Only the convenience wrappers bridge to an executable result.
    assert type(result) is CompilationResult
    assert type(result.output) is SCNativeProgram


def test_existing_qasm_source_still_rejects_a_filename_with_a_target():
    with pytest.raises(ValueError, match="filename cannot be supplied"):
        compile_qasm_to_sc(
            QasmSource(TRIANGLE_QASM),
            SCTarget(num_qubits=3, couplings=LINE),
            filename="triangle.qasm",
        )


@pytest.mark.parametrize(
    ("num_qubits", "capacity", "couplings"),
    ((1, 1, ()), (3, 3, LINE), (3, 3, FULL), (3, 5, FIVE_SITE_LINE)),
    ids=("single-site", "line", "full", "wider-line"),
)
@pytest.mark.parametrize("seed", (0, 7, 19))
def test_target_and_equivalent_simulator_compile_identically(
    num_qubits, capacity, couplings, seed
):
    source = _logical_circuit(num_qubits)
    target = SCTarget(num_qubits=capacity, couplings=couplings)
    backend = SCQubitSimulator(num_qubits=capacity, couplings=couplings)

    actual = compile_to_sc(source, target, seed=seed)
    positional = compile_to_sc(source, backend, seed=seed)
    keyword = compile_to_sc(source, backend=backend, seed=seed)

    assert actual.output == positional.output == keyword.output
    assert actual.route == positional.route == LOGICAL_ROUTE
    for result in (actual, positional):
        assert result.program.classical_registers == source.classical_registers
        assert all(
            compiled is original
            for compiled, original in zip(
                result.output.classical_registers, source.classical_registers
            )
        )


def _qasm_declaration_facts(native):
    # Each parse creates fresh registers, so compare declaration positions.
    quantum = []
    for ref, _site in native.initial_layout:
        if all(ref.register is not register for register in quantum):
            quantum.append(ref.register)
    classical = native.classical_registers

    def position(ref, registers):
        index = next(i for i, reg in enumerate(registers) if reg is ref.register)
        return index, ref.index

    operations = []
    for instruction in native.operations:
        if isinstance(instruction, NativeGate):
            operations.append(
                (
                    instruction.operation,
                    instruction.sites,
                    instruction.origin_ids,
                    instruction.generated_by,
                )
            )
        elif isinstance(instruction, NativeMeasure):
            operations.append(
                (
                    "measure",
                    instruction.site,
                    position(instruction.clbit, classical),
                    instruction.origin_ids,
                )
            )
        else:
            operations.append(("reset", instruction.site, instruction.origin_ids))
    layouts = tuple(
        tuple((position(ref, quantum), site) for ref, site in layout)
        for layout in (native.initial_layout, native.final_layout)
    )
    registers = tuple((register.name, register.size) for register in classical)
    return tuple(operations), layouts, registers


def test_qasm_target_matches_simulator_by_declaration_position():
    source = QasmSource(TRIANGLE_QASM)
    backend = SCQubitSimulator(num_qubits=3, couplings=LINE)
    target = SCTarget(num_qubits=3, couplings=LINE)

    legacy = compile_qasm_to_sc(source, backend=backend, seed=7)
    actual = compile_qasm_to_sc(source, target, seed=7)

    assert actual.route == legacy.route
    assert _qasm_declaration_facts(actual.output) == _qasm_declaration_facts(
        legacy.output
    )


def test_target_compilation_runs_on_a_compatible_simulator():
    compiled = compile_qasm_to_sc(
        TRIANGLE_QASM, SCTarget(num_qubits=3, couplings=LINE), seed=5
    )
    backend = SCQubitSimulator(num_qubits=3, couplings=LINE, runtime="numpy")

    counts = (
        backend.run(compiled, shots=64, simulation_config={"seed": 9})
        .result()
        .get_counts()
    )

    assert counts == {"110": 64}


def test_executor_and_bridge_reject_a_target():
    target = SCTarget(num_qubits=3, couplings=LINE)
    backend = SCQubitSimulator(num_qubits=3, couplings=LINE, runtime="numpy")

    with pytest.raises(TypeError):
        backend.run(target)
    with pytest.raises(TypeError, match="native must be SCNativeProgram"):
        to_sc_simulator_program(target)


@pytest.mark.parametrize(
    ("compile_with", "backend"),
    (
        (
            _compile_qasm_to_sc_rotation,
            SCTarget(num_qubits=3, couplings=LINE),
        ),
        (
            compile_qasm_to_sc,
            _SCQubitRotationSimulator(num_qubits=3, couplings=LINE),
        ),
    ),
    ids=("rotation-rejects-target", "canonical-rejects-rotation"),
)
def test_canonical_and_rotation_routes_do_not_accept_each_others_target(
    compile_with, backend
):
    with pytest.raises(PassError) as error:
        compile_with(TRIANGLE_QASM, backend)

    assert type(error.value.__cause__) is TypeError
