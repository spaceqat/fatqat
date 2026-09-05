import pytest

from fatqat.compiler import CompilationResult, PipelineNotFoundError, SC_PIPELINE
from fatqat.compiler.core import CompileContext
from fatqat.compiler.pipelines import compile_qasm_to_sc, create_sc_pipeline
from fatqat.compiler.dialects import (
    LogicalProgram,
    QasmSource,
    SCNativeProgram,
    SCProgram,
)
from fatqat.simulator import SCQubitSimulator

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

    assert isinstance(result, CompilationResult)
    assert type(result.output) is SCNativeProgram
    assert result.route == ("parse-qasm", "normalize-sc", "lower-sc-to-native")


def test_emit_stops_qasm_to_sc_pipeline_at_logical_boundary():
    result = compile_qasm_to_sc(
        _BELL_QASM,
        SCQubitSimulator(),
        emit=LogicalProgram.IR_ID,
    )

    assert isinstance(result.output, LogicalProgram)
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
