from fatqat.compiler import CompilationResult
from fatqat.compiler.pipelines import compile_qasm_to_sc, create_sc_pipeline
from fatqat.compiler.dialects import LogicalProgram, QasmSource, SCProgram

_BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def test_explicit_qasm_to_sc_pipeline_runs_both_passes():
    result = compile_qasm_to_sc(_BELL_QASM)

    assert isinstance(result, CompilationResult)
    assert isinstance(result.output, SCProgram)
    assert result.route == ("parse-qasm", "normalize-sc")


def test_emit_stops_qasm_to_sc_pipeline_at_logical_boundary():
    result = compile_qasm_to_sc(_BELL_QASM, emit=LogicalProgram.IR_ID)

    assert isinstance(result.output, LogicalProgram)
    assert result.route == ("parse-qasm",)


def test_pipeline_can_emit_the_unchanged_qasm_input_boundary():
    result = compile_qasm_to_sc(_BELL_QASM, emit=QasmSource.IR_ID)

    assert isinstance(result.output, QasmSource)
    assert result.route == ()


def test_foundation_compiler_registers_only_the_explicit_sc_pipeline():
    compiler = create_sc_pipeline()
    source = QasmSource(_BELL_QASM)

    result = compiler.compile(source, pipeline="qasm-to-sc")

    assert isinstance(result.output, SCProgram)
