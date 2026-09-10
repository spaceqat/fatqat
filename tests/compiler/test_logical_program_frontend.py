import math
import subprocess
import sys

import pytest

import fatqat as fq


def test_top_level_fatqat_loads_the_compiler_only_when_requested():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import fatqat as fq; "
                "assert 'fatqat.compiler' not in sys.modules; "
                "assert 'compiler' in dir(fq); "
                "assert fq.compiler.__name__ == 'fatqat.compiler'; "
                "assert 'fatqat.compiler' in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_logical_program_builds_a_circuit_with_gate_helpers():
    program = fq.LogicalProgram(2, 2)

    assert isinstance(program, fq.Program)
    assert program.h(0) is program
    assert program.cx(0, 1) is program
    assert program.measure_all() is program

    dag = program.dag()
    assert tuple(node.name for node in dag.nodes) == (
        "H",
        "CX",
        "Measurement",
    )
    assert tuple(ref.index for ref in dag.nodes[1].targets) == (0, 1)
    assert tuple(ref.index for ref in dag.nodes[2].outputs) == (0, 1)


def test_logical_program_accepts_conditions_and_runs_on_the_general_simulator():
    program = fq.LogicalProgram(2, 2)
    program.x(0)
    program.measure(0, 0)
    program.add(fq.operations.X, 1, condition=(0, 1))
    program.measure(1, 1)

    counts = (
        fq.simulator.Simulator("SV", runtime="numpy")
        .run(program, shots=8, simulation_config={"seed": 3})
        .result()
        .get_counts_as_tuples()
    )

    assert counts == {(1, 1): 8}


def test_static_compiler_reports_a_logical_program_condition():
    from fatqat.compiler import PassError, compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    program = fq.LogicalProgram(1, 1)
    program.add(fq.operations.X, 0, condition=(0, 0))

    with pytest.raises(PassError, match="classical condition is not supported"):
        compile_to_sc(
            program,
            fq.simulator.SCQubitSimulator(),
            emit=LogicalIR.IR_ID,
        )


@pytest.mark.parametrize(
    ("operation", "targets"),
    [
        (fq.operations.Put, (0, 1)),
        (fq.operations.Pair, (0, 1)),
        (fq.operations.Unpair, (0, 1)),
    ],
)
def test_logical_program_rejects_device_operations(operation, targets):
    program = fq.LogicalProgram(2)

    with pytest.raises(ValueError, match="not a logical operation"):
        program.add(operation, targets)


def test_logical_program_views_are_expanded_when_frozen():
    from fatqat.compiler import compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    qubits = fq.QuantumRegister(2, name="q")
    program = fq.LogicalProgram([qubits])
    program.h(qubits.all())

    logical = compile_to_sc(
        program,
        fq.simulator.SCQubitSimulator(),
        emit=LogicalIR.IR_ID,
    ).output

    assert tuple(item.operation.name for item in logical.instructions) == ("H", "H")
    assert tuple(item.operands for item in logical.instructions) == (
        (qubits[0],),
        (qubits[1],),
    )


@pytest.mark.parametrize(
    ("method", "args", "operation_name"),
    [
        ("i", (0,), "I"),
        ("h", (0,), "H"),
        ("x", (0,), "X"),
        ("y", (0,), "Y"),
        ("z", (0,), "Z"),
        ("s", (0,), "S"),
        ("sdg", (0,), "Sdg"),
        ("t", (0,), "T"),
        ("tdg", (0,), "Tdg"),
        ("sx", (0,), "SX"),
        ("rx", (0.1, 0), "RX"),
        ("ry", (0.2, 0), "RY"),
        ("rz", (0.3, 0), "RZ"),
        ("phase", (0.4, 0), "Phase"),
        ("cx", (0, 1), "CX"),
        ("cz", (0, 1), "CZ"),
        ("swap", (0, 1), "Swap"),
        ("reset", (0,), "Reset"),
    ],
)
def test_logical_gate_helpers_append_the_named_operation(method, args, operation_name):
    program = fq.LogicalProgram(2)

    returned = getattr(program, method)(*args)

    assert returned is program
    assert tuple(node.name for node in program.dag().nodes) == (operation_name,)


def test_logical_program_add_accepts_separate_or_grouped_operands():
    program = fq.LogicalProgram(2)

    program.add(fq.operations.H, 0)
    program.add(fq.operations.CX, 0, 1)
    program.add(fq.operations.CZ, (1, 0))

    assert tuple(node.name for node in program.dag().nodes) == ("H", "CX", "CZ")
    assert tuple(ref.index for ref in program.dag().nodes[2].targets) == (1, 0)


def test_logical_program_uses_explicit_register_refs_and_preserves_metadata():
    qreg = fq.QuantumRegister(2, name="data")
    creg = fq.ClassicalRegister(1, name="out")
    program = fq.LogicalProgram([qreg], [creg], metadata={"label": "bell"})

    program.rx(math.pi / 4, qreg[1]).measure(qreg[1], creg[0])

    assert program.quantum_registers == (qreg,)
    assert program.classical_registers == (creg,)
    assert program.metadata == {"label": "bell"}
    assert program.dag().nodes[0].targets == (qreg[1],)
    assert program.dag().nodes[1].outputs == (creg[0],)


def test_logical_program_copy_can_be_edited_independently():
    original = fq.LogicalProgram(2, metadata={"branch": "original"}).h(0)

    copied = original.copy()
    copied.x(1)
    copied.metadata["branch"] = "copy"

    assert type(copied) is fq.LogicalProgram
    assert tuple(node.name for node in original.dag().nodes) == ("H",)
    assert tuple(node.name for node in copied.dag().nodes) == ("H", "X")
    assert original.metadata == {"branch": "original"}
    assert copied.metadata == {"branch": "copy"}


def test_logical_program_parameters_can_be_bound_before_compilation():
    from fatqat.compiler import ValidationError, compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    theta = fq.Parameter("theta")
    template = fq.LogicalProgram(1).rx(theta, 0)

    bound = template.assign_parameters({theta: 0.25})
    logical = compile_to_sc(
        bound,
        fq.simulator.SCQubitSimulator(),
        emit=LogicalIR.IR_ID,
    ).output

    assert bound is not template
    assert type(bound) is fq.LogicalProgram
    assert logical.instructions[0].operation.theta == 0.25
    with pytest.raises(ValidationError, match="finite real number"):
        compile_to_sc(
            template,
            fq.simulator.SCQubitSimulator(),
            emit=LogicalIR.IR_ID,
        )


def test_logical_program_rejects_a_foreign_register_ref():
    program = fq.LogicalProgram(1)
    foreign = fq.QuantumRegister(1, name="foreign")[0]

    with pytest.raises(ValueError, match="does not belong"):
        program.h(foreign)


_BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def _bell_program():
    return fq.LogicalProgram(2, 2).h(0).cx(0, 1).measure_all()


def test_compile_to_sc_runs_the_logical_frontend_route():
    from fatqat.compiler import compile_to_sc
    from fatqat.compiler.dialects import SCNativeProgram

    result = compile_to_sc(_bell_program(), fq.simulator.SCQubitSimulator())

    assert type(result.output) is SCNativeProgram
    assert result.route == (
        "freeze-logical",
        "normalize-sc",
        "lower-sc-to-native",
    )


def test_compile_to_sc_rejects_a_general_program():
    from fatqat.compiler import ValidationError, compile_to_sc

    program = fq.Program(1)
    program.add(fq.operations.H, 0)

    with pytest.raises(ValidationError, match="expects LogicalProgram"):
        compile_to_sc(program, fq.simulator.SCQubitSimulator())


def test_logical_freeze_is_deterministic_and_does_not_edit_the_source():
    from fatqat.compiler import compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    program = _bell_program()
    before = tuple(node.name for node in program.dag().nodes)

    first = compile_to_sc(
        program,
        fq.simulator.SCQubitSimulator(),
        emit=LogicalIR.IR_ID,
    )
    second = compile_to_sc(
        program,
        fq.simulator.SCQubitSimulator(),
        emit=LogicalIR.IR_ID,
    )

    assert type(first.output) is LogicalIR
    assert first.output == second.output
    assert tuple(item.operation_id for item in first.output.instructions) == (
        "logical.0",
        "logical.1",
        "logical.2",
        "logical.3",
    )
    assert tuple(node.name for node in program.dag().nodes) == before
    assert first.route == ("freeze-logical",)


def test_python_and_qasm_frontends_produce_the_same_logical_ir():
    from fatqat.compiler import compile_qasm_to_sc, compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    backend = fq.simulator.SCQubitSimulator()
    python_ir = compile_to_sc(_bell_program(), backend, emit=LogicalIR.IR_ID).output
    qasm_ir = compile_qasm_to_sc(_BELL_QASM, backend, emit=LogicalIR.IR_ID).output

    def semantic_facts(logical):
        facts = []
        for item in logical.instructions:
            operands = item.operands if hasattr(item, "operands") else (item.qubit,)
            facts.append(
                (
                    type(item).__name__,
                    getattr(getattr(item, "operation", None), "name", "Measurement"),
                    tuple(ref.index for ref in operands),
                    getattr(getattr(item, "clbit", None), "index", None),
                )
            )
        return tuple(facts)

    expected = (
        ("LogicalGate", "H", (0,), None),
        ("LogicalGate", "CX", (0, 1), None),
        ("LogicalMeasure", "Measurement", (0,), 0),
        ("LogicalMeasure", "Measurement", (1,), 1),
    )
    assert semantic_facts(python_ir) == expected
    assert semantic_facts(qasm_ir) == expected


def test_compile_to_na_reuses_the_existing_normalization_and_zap_route():
    from fatqat.compiler import compile_to_na
    from fatqat.compiler.algorithms.zap import load_architecture
    from fatqat.compiler.dialects import ZonedPlan

    result = compile_to_na(_bell_program(), load_architecture("default"))

    assert type(result.output) is ZonedPlan
    assert result.route == (
        "freeze-logical",
        "normalize-na",
        "schedule-with-zap",
    )


def test_compile_to_na_reports_a_target_specific_unsupported_gate():
    from fatqat.compiler import PassError, compile_to_na
    from fatqat.compiler.algorithms.zap import load_architecture
    from fatqat.compiler.dialects import NAProgram

    with pytest.raises(PassError, match="SX.*not supported"):
        compile_to_na(
            fq.LogicalProgram(1).sx(0),
            load_architecture("default"),
            emit=NAProgram.IR_ID,
        )


def test_logical_program_runs_end_to_end_on_the_sc_simulator():
    backend = fq.simulator.SCQubitSimulator(
        num_qubits=2,
        couplings=((0, 1),),
        runtime="numpy",
    )
    compiled = fq.compiler.compile_to_sc(_bell_program(), backend)

    counts = (
        backend.run(
            compiled,
            shots=32,
            simulation_config={"seed": 3},
        )
        .result()
        .get_counts()
    )

    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 32


def test_logical_program_runs_end_to_end_on_the_na_simulator():
    from fatqat.compiler.algorithms.zap import load_architecture

    compiled = fq.compiler.compile_to_na(_bell_program(), load_architecture("default"))
    backend = fq.simulator.AtomArraySimulator(runtime="numpy")

    counts = (
        backend.run(
            compiled,
            shots=32,
            simulation_config={"seed": 3},
        )
        .result()
        .get_counts()
    )

    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 32
