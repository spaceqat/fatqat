import math
import subprocess
import sys
from typing import get_type_hints

import pytest

import fatqat as fq
import fatqat.operations as ops


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


def test_logical_program_builds_a_circuit():
    program = fq.LogicalProgram(2, 2)

    assert program.add(ops.H, 0) is None
    assert program.add(ops.CX, (0, 1)) is None
    assert program.measure_all() is None

    dag = program.dag()
    assert tuple(node.name for node in dag.nodes) == ("H", "CX", "Measurement")
    assert tuple(ref.index for ref in dag.nodes[1].targets) == (0, 1)
    assert tuple(ref.index for ref in dag.nodes[2].outputs) == (0, 1)


@pytest.mark.parametrize(
    ("operation", "targets", "operation_name"),
    [
        (ops.I, 0, "I"),
        (ops.H, 0, "H"),
        (ops.X, 0, "X"),
        (ops.Y, 0, "Y"),
        (ops.Z, 0, "Z"),
        (ops.S, 0, "S"),
        (ops.Sdg, 0, "Sdg"),
        (ops.T, 0, "T"),
        (ops.Tdg, 0, "Tdg"),
        (ops.SX, 0, "SX"),
        (ops.RX(0.1), 0, "RX"),
        (ops.RY(0.2), 0, "RY"),
        (ops.RZ(0.3), 0, "RZ"),
        (ops.Phase(0.4), 0, "Phase"),
        (ops.CX, (0, 1), "CX"),
        (ops.CZ, (0, 1), "CZ"),
        (ops.Swap, (0, 1), "Swap"),
        (ops.Reset, 0, "Reset"),
        (ops.Barrier, (0, 1), "Barrier"),
        (ops.U(0.1, 0.2, 0.3), 0, "U"),
    ],
)
def test_logical_program_add_appends_the_named_operation(
    operation, targets, operation_name
):
    program = fq.LogicalProgram(2)
    assert program.add(operation, targets) is None
    assert tuple(node.name for node in program.dag().nodes) == (operation_name,)


def test_logical_program_uses_explicit_register_refs_and_preserves_metadata():
    qreg = fq.QuantumRegister(2, name="data")
    creg = fq.ClassicalRegister(1, name="out")
    program = fq.LogicalProgram([qreg], [creg], metadata={"label": "bell"})

    program.add(ops.RX(math.pi / 4), qreg[1])
    program.measure(qreg[1], creg[0])

    assert isinstance(program, fq.Program)
    assert program.quantum_registers == (qreg,)
    assert program.classical_registers == (creg,)
    assert program.metadata == {"label": "bell"}
    assert program.dag().nodes[0].targets == (qreg[1],)
    assert program.dag().nodes[1].outputs == (creg[0],)


def test_logical_program_copy_can_be_edited_independently():
    original = fq.LogicalProgram(2, metadata={"branch": "original"})
    original.add(ops.H, 0)

    copied = original.copy()
    copied.add(ops.X, 1)
    copied.metadata["branch"] = "copy"

    with pytest.raises(ValueError, match="not a logical operation"):
        copied.add(ops.Pair, (0, 1))

    assert type(copied) is fq.LogicalProgram
    assert tuple(node.name for node in original.dag().nodes) == ("H",)
    assert tuple(node.name for node in copied.dag().nodes) == ("H", "X")
    assert original.metadata == {"branch": "original"}
    assert copied.metadata == {"branch": "copy"}


def test_logical_program_and_compiler_entry_points_preserve_public_types():
    from fatqat.compiler import compile_to_na, compile_to_sc

    assert get_type_hints(fq.LogicalProgram.copy)["return"] is fq.LogicalProgram
    assert (
        get_type_hints(fq.LogicalProgram.assign_parameters)["return"]
        is fq.LogicalProgram
    )
    assert get_type_hints(compile_to_sc)["source"] is fq.LogicalProgram
    assert get_type_hints(compile_to_na)["source"] is fq.LogicalProgram


def test_logical_program_parameters_can_be_bound_before_compilation():
    from fatqat.compiler import ValidationError, compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    backend = fq.simulator.SCQubitSimulator()

    theta = fq.Parameter("theta")
    template = fq.LogicalProgram(1)
    template.add(ops.RX(theta), 0)

    bound = template.assign_parameters({theta: 0.25})
    logical = compile_to_sc(bound, backend, emit=LogicalIR.IR_ID).output

    assert type(bound) is fq.LogicalProgram
    assert bound is not template
    assert logical.instructions[0].operation.theta == 0.25
    with pytest.raises(ValidationError, match="finite real number"):
        compile_to_sc(template, backend, emit=LogicalIR.IR_ID)


def test_logical_program_rejects_a_foreign_register_ref():
    program = fq.LogicalProgram(1)
    foreign = fq.QuantumRegister(1, name="foreign")[0]

    with pytest.raises(ValueError, match="does not belong"):
        program.add(ops.H, foreign)


_BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def _bell_program():
    program = fq.LogicalProgram(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure_all()
    return program


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

    program.add(ops.X, 0)
    assert len(first.output.instructions) == 4


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


@pytest.mark.parametrize("operation", [ops.SX, ops.Reset])
def test_compile_to_na_reports_a_target_specific_unsupported_gate(
    operation,
):
    from fatqat.compiler import PassError, compile_to_na
    from fatqat.compiler.algorithms.zap import load_architecture
    from fatqat.compiler.dialects import NAProgram

    program = fq.LogicalProgram(1)
    program.add(operation, 0)
    with pytest.raises(PassError, match=f"{operation.name}.*not supported"):
        compile_to_na(
            program,
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
    assert compiled.route == ("freeze-logical", "normalize-sc", "lower-sc-to-native")

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
    assert compiled.route == ("freeze-logical", "normalize-na", "schedule-with-zap")
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


@pytest.mark.parametrize("target", ["sc", "na"])
def test_compiler_rejects_plain_program(target):
    from fatqat.compiler import ValidationError, compile_to_na, compile_to_sc
    from fatqat.compiler.algorithms.zap import load_architecture

    program = fq.Program(1)
    program.add(ops.H, 0)

    with pytest.raises(ValidationError, match="expects LogicalProgram, got Program"):
        if target == "sc":
            compile_to_sc(program, fq.simulator.SCQubitSimulator())
        else:
            compile_to_na(program, load_architecture("default"))


class CustomRX(ops.RX):
    """A custom subclass must not bypass the built-in operation restriction."""


@pytest.mark.parametrize(
    ("operation", "targets"),
    [
        (ops.Put, (0, 1)),
        (ops.Pair, (0, 1)),
        (ops.Unpair, (0, 1)),
        (CustomRX(0.25), 0),
    ],
)
def test_logical_program_rejects_nonlogical_operations_without_mutation(
    operation, targets
):
    program = fq.LogicalProgram(2)
    program.add(ops.H, 0)

    with pytest.raises(ValueError, match="not a logical operation"):
        program.add(operation, targets)

    assert tuple(node.name for node in program.dag().nodes) == ("H",)


def test_logical_program_rejects_direct_pulse_controls():
    model = fq.emulator.TransmonModel.from_document(
        fq.emulator.load_model_document("transmon.reference")
    )
    waveform = fq.emulator.SampledWaveform((0.0, 1.0), (0.0, 0.0))
    control = fq.emulator.PulseControl(model.control.drive("q0"), waveform)
    program = fq.LogicalProgram(1)

    with pytest.raises(ValueError, match="PulseOperation is not a logical operation"):
        program.add(ops.PulseOperation(1.0, (control,)))

    assert not program.dag().nodes


@pytest.mark.parametrize("operation", [ops.RX, ops.Measurement])
def test_logical_program_preserves_invalid_operation_guidance(operation):
    program = fq.LogicalProgram(1)

    with pytest.raises(TypeError):
        program.add(operation, 0)

    assert not program.dag().nodes


def test_conditions_run_directly_but_are_rejected_by_the_static_compiler():
    from fatqat.compiler import PassError, compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    program = fq.LogicalProgram(2, 2)
    program.add(ops.X, 0)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    program.measure(1, 1)

    counts = (
        fq.simulator.Simulator("SV", runtime="numpy")
        .run(program, shots=8, simulation_config={"seed": 3})
        .result()
        .get_counts_as_tuples()
    )
    assert counts == {(1, 1): 8}

    with pytest.raises(PassError, match="classical condition is not supported"):
        compile_to_sc(program, fq.simulator.SCQubitSimulator(), emit=LogicalIR.IR_ID)


def test_register_views_compile_like_scalar_gates():
    from fatqat.compiler import compile_to_sc
    from fatqat.compiler.dialects import LogicalIR

    first = fq.QuantumRegister(2, name="first")
    second = fq.QuantumRegister(2, name="second")
    grouped = fq.LogicalProgram([first, second])
    grouped.add(ops.H, first.all())
    grouped.add(ops.CX, (first.all(), second.all()))
    scalar = fq.LogicalProgram([first, second])
    for index in range(2):
        scalar.add(ops.H, first[index])
    for index in range(2):
        scalar.add(ops.CX, (first[index], second[index]))

    backend = fq.simulator.SCQubitSimulator()
    expanded = compile_to_sc(grouped, backend, emit=LogicalIR.IR_ID).output
    expected = compile_to_sc(scalar, backend, emit=LogicalIR.IR_ID).output

    assert expanded == expected
    assert tuple(item.operands for item in expanded.instructions) == (
        (first[0],),
        (first[1],),
        (first[0], second[0]),
        (first[1], second[1]),
    )


def test_logical_program_accepts_builtin_qudit_operations():
    register = fq.QuantumRegister(2, dim=3)
    program = fq.LogicalProgram([register])
    program.add(ops.Fourier, register[0])
    program.add(ops.Sum, (register[0], register[1]))

    assert tuple(node.name for node in program.dag().nodes) == ("Fourier", "Sum")
