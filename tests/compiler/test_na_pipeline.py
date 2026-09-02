from fatqat.compiler import CompileContext
from fatqat.compiler.algorithms.zap import load_architecture
from fatqat.compiler.dialects import NAProgram, QasmSource, ZonedPlan
import fatqat.compiler.passes.na_zap as na_zap

_BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def test_na_pipeline_runs_the_explicit_qasm_normalize_zap_route():
    from fatqat.compiler import NA_PIPELINE, create_na_pipeline

    result = create_na_pipeline().compile(
        QasmSource(_BELL_QASM),
        pipeline=NA_PIPELINE,
        context=CompileContext(target=load_architecture("default")),
    )

    assert isinstance(result.output, ZonedPlan)
    assert result.route == ("parse-qasm", "normalize-na", "schedule-with-zap")


def test_na_emit_boundary_does_not_call_internal_zap(monkeypatch):
    from fatqat.compiler import NA_PIPELINE, create_na_pipeline

    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("ZAP must not run before the NA emit boundary")

    monkeypatch.setattr(na_zap, "compile_interactions", fail_if_called)
    result = create_na_pipeline().compile(
        QasmSource(_BELL_QASM),
        pipeline=NA_PIPELINE,
        emit=NAProgram.IR_ID,
        context=CompileContext(target=load_architecture("default")),
    )

    assert isinstance(result.output, NAProgram)
    assert result.route == ("parse-qasm", "normalize-na")
    assert not called


def test_public_na_api_can_emit_the_normalized_boundary_without_scheduling():
    from fatqat.compiler import compile_qasm_to_na

    result = compile_qasm_to_na(
        _BELL_QASM,
        load_architecture("default"),
        emit=NAProgram.IR_ID,
    )

    assert isinstance(result.output, NAProgram)
    assert result.route == ("parse-qasm", "normalize-na")
