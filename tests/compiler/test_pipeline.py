import pytest

from fatqat.compiler import (
    CompilationResult,
    ExecutableCompilationResult,
    PassError,
    PipelineNotFoundError,
    SC_PIPELINE,
    SCTarget,
    ValidationError,
)
from fatqat.compiler.core import CompileContext
from fatqat.compiler.pipelines import compile_qasm_to_sc, create_sc_pipeline
from fatqat.compiler.dialects import (
    LogicalIR,
    QasmSource,
    SCNativeProgram,
    SCProgram,
)
from fatqat.simulator import SCQubitSimulator
from fatqat.execution import ExecutableProgram
from fatqat.operations import SX

_BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def test_explicit_qasm_to_sc_pipeline_runs_the_canonical_route():
    result = compile_qasm_to_sc(_BELL_QASM, SCQubitSimulator())

    assert isinstance(result, ExecutableCompilationResult)
    assert type(result.output) is SCNativeProgram
    assert result.route == ("parse-qasm", "normalize-sc", "lower-sc-to-native")


def test_emit_stops_qasm_to_sc_pipeline_at_logical_boundary():
    result = compile_qasm_to_sc(
        _BELL_QASM,
        SCQubitSimulator(),
        emit=LogicalIR.IR_ID,
    )

    assert isinstance(result.output, LogicalIR)
    assert type(result) is CompilationResult
    assert not isinstance(result, ExecutableProgram)
    assert result.route == ("parse-qasm",)


def test_pipeline_can_emit_the_unchanged_qasm_input_boundary():
    result = compile_qasm_to_sc(
        _BELL_QASM,
        SCQubitSimulator(),
        emit=QasmSource.IR_ID,
        filename="bell.qasm",
    )

    assert isinstance(result.output, QasmSource)
    assert result.output.filename == "bell.qasm"
    assert result.route == ()


def test_sc_compiler_registers_the_public_pipeline_and_emit_boundaries():
    compiler = create_sc_pipeline()
    source = QasmSource(_BELL_QASM)
    backend = SCQubitSimulator()
    context = CompileContext(target=backend, options={"seed": 3})

    assert SC_PIPELINE == "qasm-to-sc"
    with pytest.raises(PipelineNotFoundError, match="unknown pipeline"):
        compiler.compile(
            source,
            pipeline="qasm-to-sc-rotation",
            context=context,
        )

    native = compiler.compile(source, pipeline=SC_PIPELINE, context=context)
    intermediate = compiler.compile(
        source,
        pipeline=SC_PIPELINE,
        emit=SCProgram.IR_ID,
        context=context,
    )

    assert type(native.output) is SCNativeProgram
    assert type(intermediate.output) is SCProgram
    assert intermediate.route == ("parse-qasm", "normalize-sc")


def test_fixed_seed_repeats_full_pipeline_output():
    backend = SCQubitSimulator()

    first = compile_qasm_to_sc(_BELL_QASM, backend, seed=7)
    second = compile_qasm_to_sc(_BELL_QASM, backend, seed=7)

    def observable_facts(result):
        native = result.output
        operations = tuple(
            (
                type(instruction).__name__,
                getattr(instruction, "operation", None),
                getattr(instruction, "sites", None),
                getattr(instruction, "site", None),
                getattr(instruction, "origin_ids"),
                getattr(instruction, "generated_by", None),
                getattr(getattr(instruction, "clbit", None), "index", None),
            )
            for instruction in native.operations
        )
        layouts = tuple(
            tuple(
                (logical.register.name, logical.index, site) for logical, site in layout
            )
            for layout in (native.initial_layout, native.final_layout)
        )
        return result.route, operations, layouts

    assert observable_facts(second) == observable_facts(first)


class _UnreadableSCSimulator(SCQubitSimulator):
    @property
    def implementation_map(self):
        raise RuntimeError("constraint-read")


class _NoSXSimulator(SCQubitSimulator):
    @property
    def implementation_map(self):
        result = super().implementation_map
        result.remove(SX)
        return result


_BACKENDS = {
    "object": object,
    "unreadable": lambda: _UnreadableSCSimulator(num_qubits=2, couplings=((0, 1),)),
    "target": lambda: SCTarget(num_qubits=2, couplings=((0, 1),)),
    "small-target": lambda: SCTarget(num_qubits=1, couplings=()),
    "missing-sx": lambda: _NoSXSimulator(num_qubits=2, couplings=((0, 1),)),
}


@pytest.mark.parametrize("backend_kind", ("object", "unreadable", "target"))
@pytest.mark.parametrize(
    ("boundary", "route"),
    (
        (QasmSource.IR_ID, ()),
        (LogicalIR.IR_ID, ("parse-qasm",)),
        (SCProgram.IR_ID, ("parse-qasm", "normalize-sc")),
    ),
    ids=("source", "logical-ir", "sc-program"),
)
@pytest.mark.parametrize("as_text", (False, True), ids=("qasm-source", "text"))
def test_qasm_early_emit_does_not_read_the_backend(
    backend_kind, boundary, route, as_text
):
    source = _BELL_QASM if as_text else QasmSource(_BELL_QASM)

    result = compile_qasm_to_sc(source, _BACKENDS[backend_kind](), emit=boundary)

    assert type(result) is CompilationResult
    assert result.route == route
    assert type(result.output).IR_ID == boundary
    if not route and not as_text:
        assert result.output is source


@pytest.mark.parametrize(
    ("backend_kind", "cause_type", "message"),
    (
        ("object", TypeError, "SC lowering requires SCTarget or SCQubitSimulator"),
        ("unreadable", RuntimeError, "constraint-read"),
        ("small-target", ValueError, "not enough physical sites for SC program qubits"),
        ("missing-sx", ValidationError, "native operation SX is illegal"),
    ),
)
def test_qasm_native_failures_keep_the_lowering_stage_and_cause(
    backend_kind, cause_type, message
):
    with pytest.raises(PassError) as error:
        compile_qasm_to_sc(_BELL_QASM, _BACKENDS[backend_kind]())

    assert error.value.pass_name == "lower-sc-to-native"
    assert type(error.value.__cause__) is cause_type
    assert str(error.value.__cause__).startswith(message)


def test_qasm_parse_errors_stay_in_the_parse_stage_for_a_target():
    with pytest.raises(PassError) as error:
        compile_qasm_to_sc(
            "OPENQASM 3.0; qubit[1] q; bogus q[0];",
            SCTarget(num_qubits=1, couplings=()),
        )

    assert error.value.pass_name == "parse-qasm"
