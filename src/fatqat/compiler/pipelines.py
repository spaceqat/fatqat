"""Construct and run the compiler's explicit translation pipelines."""

from __future__ import annotations

from collections.abc import Mapping

from ..simulator import SCQubitSimulator
from .core import CompilationResult, CompileContext, Compiler, Pipeline
from .dialects.logical_gate import LogicalProgram, verify_logical_program
from .dialects.na_gate import NAProgram, verify_na_program
from .dialects.na_zoned import ZonedPlan, verify_zoned_plan
from .dialects.qasm import QasmSource, verify_qasm_source
from .dialects.sc_gate import SCProgram, verify_sc_program
from .dialects.sc_native import (
    SCNativeProgram,
    _RotationNativeProgram,
    _verify_rotation_native_program,
    verify_sc_native_program,
)
from .passes.qasm import parse_qasm
from .passes.na import normalize_na
from .passes.na_zap import schedule_with_zap
from .passes.sc import normalize_sc
from .passes.sc_target import _lower_sc_to_rotation, lower_sc_to_native

SC_PIPELINE = "qasm-to-sc"
_SC_ROTATION_PIPELINE = "qasm-to-sc-rotation"
NA_PIPELINE = "qasm-to-na-zap"


def create_sc_pipeline() -> Compiler:
    """Create the QASM-to-native superconducting compiler."""

    compiler = Compiler()
    compiler.register_ir(QasmSource, verify_qasm_source)
    compiler.register_ir(LogicalProgram, verify_logical_program)
    compiler.register_ir(SCProgram, verify_sc_program)
    compiler.register_ir(SCNativeProgram, verify_sc_native_program)
    compiler.register_pipeline(
        Pipeline(SC_PIPELINE, (parse_qasm, normalize_sc, lower_sc_to_native))
    )
    return compiler


def _create_sc_rotation_pipeline() -> Compiler:
    """Create the private QASM-to-rotation-native compiler."""

    compiler = create_sc_pipeline()
    compiler.register_ir(_RotationNativeProgram, _verify_rotation_native_program)
    compiler.register_pipeline(
        Pipeline(
            _SC_ROTATION_PIPELINE,
            (parse_qasm, normalize_sc, _lower_sc_to_rotation),
        )
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
    backend: SCQubitSimulator,
    *,
    emit: str = SCNativeProgram.IR_ID,
    filename: str | None = None,
    seed: int = 0,
) -> CompilationResult:
    """Compile static numeric OpenQASM to canonical SC native IR."""

    return create_sc_pipeline().compile(
        _qasm_source(source, filename),
        pipeline=SC_PIPELINE,
        emit=emit,
        context=CompileContext(target=backend, options={"seed": seed}),
    )


def _compile_qasm_to_sc_rotation(
    source: str | QasmSource,
    backend: object,
    *,
    emit: str = _RotationNativeProgram.IR_ID,
    filename: str | None = None,
    seed: int = 0,
) -> CompilationResult:
    """Compile static numeric OpenQASM to private rotation-native IR."""

    return _create_sc_rotation_pipeline().compile(
        _qasm_source(source, filename),
        pipeline=_SC_ROTATION_PIPELINE,
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
