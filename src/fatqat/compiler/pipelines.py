"""Construct and run the compiler's explicit translation pipelines."""

from __future__ import annotations

from collections.abc import Mapping

from .core import CompilationResult, CompileContext, Compiler, Pipeline
from .dialects.logical_gate import LogicalProgram, verify_logical_program
from .dialects.na_gate import NAProgram, verify_na_program
from .dialects.na_zoned import ZonedPlan, verify_zoned_plan
from .dialects.qasm import QasmSource, verify_qasm_source
from .dialects.sc_gate import SCProgram, verify_sc_program
from .dialects.sc_native import (
    GoogleProgram,
    IBMProgram,
    verify_google_program,
    verify_ibm_program,
)
from .passes.qasm import parse_qasm
from .passes.na import normalize_na
from .passes.na_zap import schedule_with_zap
from .passes.sc import normalize_sc
from .passes.sc_target import lower_sc_to_google, lower_sc_to_ibm

SC_FOUNDATION_PIPELINE = "qasm-to-sc"
IBM_PIPELINE = "qasm-to-ibm"
GOOGLE_PIPELINE = "qasm-to-google"
NA_PIPELINE = "qasm-to-na-zap"


def create_sc_pipeline() -> Compiler:
    """Create a compiler containing only the explicit QASM-to-SC foundation route."""

    compiler = Compiler()
    compiler.register_ir(QasmSource, verify_qasm_source)
    compiler.register_ir(LogicalProgram, verify_logical_program)
    compiler.register_ir(SCProgram, verify_sc_program)
    compiler.register_pipeline(
        Pipeline(SC_FOUNDATION_PIPELINE, (parse_qasm, normalize_sc))
    )
    return compiler


def create_ibm_pipeline() -> Compiler:
    """Create the explicit QASM-to-IBM native pipeline."""

    compiler = create_sc_pipeline()
    compiler.register_ir(IBMProgram, verify_ibm_program)
    compiler.register_pipeline(
        Pipeline(IBM_PIPELINE, (parse_qasm, normalize_sc, lower_sc_to_ibm))
    )
    return compiler


def create_google_pipeline() -> Compiler:
    """Create the explicit QASM-to-Google native pipeline."""

    compiler = create_sc_pipeline()
    compiler.register_ir(GoogleProgram, verify_google_program)
    compiler.register_pipeline(
        Pipeline(GOOGLE_PIPELINE, (parse_qasm, normalize_sc, lower_sc_to_google))
    )
    return compiler


def create_na_pipeline() -> Compiler:
    """Create the explicit QASM-to-neutral-atom ZAP scheduling pipeline."""

    compiler = Compiler()
    compiler.register_ir(QasmSource, verify_qasm_source)
    compiler.register_ir(LogicalProgram, verify_logical_program)
    compiler.register_ir(NAProgram, verify_na_program)
    compiler.register_ir(ZonedPlan, verify_zoned_plan)
    compiler.register_pipeline(
        Pipeline(NA_PIPELINE, (parse_qasm, normalize_na, schedule_with_zap))
    )
    return compiler


def _qasm_source(source: str | QasmSource, filename: str | None) -> QasmSource:
    if isinstance(source, str):
        return QasmSource(source, filename)
    if type(source) is QasmSource:
        if filename is not None:
            raise ValueError("filename cannot be supplied with an existing QasmSource")
        return source
    raise TypeError("source must be OpenQASM text or QasmSource")


def compile_qasm_to_sc(
    source: str | QasmSource,
    *,
    emit: str = SCProgram.IR_ID,
    filename: str | None = None,
) -> CompilationResult:
    """Compile numeric, static OpenQASM to the logical or unified SC boundary."""

    qasm_source = _qasm_source(source, filename)
    return create_sc_pipeline().compile(
        qasm_source,
        pipeline=SC_FOUNDATION_PIPELINE,
        emit=emit,
    )


def compile_qasm_to_ibm(
    source: str | QasmSource,
    backend: object,
    *,
    emit: str = IBMProgram.IR_ID,
    filename: str | None = None,
    seed: int = 0,
) -> CompilationResult:
    """Compile static numeric OpenQASM to IBM native physical-site IR."""

    return create_ibm_pipeline().compile(
        _qasm_source(source, filename),
        pipeline=IBM_PIPELINE,
        emit=emit,
        context=CompileContext(target=backend, options={"seed": seed}),
    )


def compile_qasm_to_google(
    source: str | QasmSource,
    backend: object,
    *,
    emit: str = GoogleProgram.IR_ID,
    filename: str | None = None,
    seed: int = 0,
) -> CompilationResult:
    """Compile static numeric OpenQASM to Google native physical-site IR."""

    return create_google_pipeline().compile(
        _qasm_source(source, filename),
        pipeline=GOOGLE_PIPELINE,
        emit=emit,
        context=CompileContext(target=backend, options={"seed": seed}),
    )


def compile_qasm_to_na(
    source: str | QasmSource,
    architecture: Mapping[str, object],
    *,
    emit: str = ZonedPlan.IR_ID,
    filename: str | None = None,
) -> CompilationResult:
    """Compile static numeric OpenQASM to a ZAP-scheduled NA physical plan."""

    return create_na_pipeline().compile(
        _qasm_source(source, filename),
        pipeline=NA_PIPELINE,
        emit=emit,
        context=CompileContext(target=architecture),
    )
