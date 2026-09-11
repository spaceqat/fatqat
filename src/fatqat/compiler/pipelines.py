"""Construct and run the compiler's explicit translation pipelines."""

from __future__ import annotations

from collections.abc import Mapping

from ..logical_program import LogicalProgram
from ..simulator import SCQubitSimulator
from .core import (
    CompilationResult,
    CompileContext,
    Compiler,
    ExecutableCompilationResult,
    Pipeline,
)
from .dialects.logical_gate import LogicalIR, verify_logical_ir
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
from .errors import ValidationError
from .passes.qasm import freeze_logical, parse_qasm
from .passes.na import normalize_na
from .passes.na_zap import schedule_with_zap
from .passes.sc import normalize_sc
from .passes.sc_target import _lower_sc_to_rotation, lower_sc_to_native
from .simulator_bridge import to_na_simulator_program, to_sc_simulator_program

SC_PIPELINE = "qasm-to-sc"
LOGICAL_SC_PIPELINE = "logical-to-sc"
_SC_ROTATION_PIPELINE = "qasm-to-sc-rotation"
NA_PIPELINE = "qasm-to-na-zap"
LOGICAL_NA_PIPELINE = "logical-to-na"


def _verify_logical_source(program: object) -> None:
    if type(program) is not LogicalProgram:
        raise ValidationError("expected LogicalProgram")


def create_sc_pipeline() -> Compiler:
    """Create the QASM and Python-frontend superconducting compiler."""

    compiler = Compiler()
    compiler.register_ir(QasmSource, verify_qasm_source)
    compiler.register_ir(LogicalProgram, _verify_logical_source)
    compiler.register_ir(LogicalIR, verify_logical_ir)
    compiler.register_ir(SCProgram, verify_sc_program)
    compiler.register_ir(SCNativeProgram, verify_sc_native_program)
    compiler.register_pipeline(
        Pipeline(SC_PIPELINE, (parse_qasm, normalize_sc, lower_sc_to_native))
    )
    compiler.register_pipeline(
        Pipeline(
            LOGICAL_SC_PIPELINE,
            (freeze_logical, normalize_sc, lower_sc_to_native),
        )
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
    """Create the QASM and Python-frontend neutral-atom compiler."""

    compiler = Compiler()
    compiler.register_ir(QasmSource, verify_qasm_source)
    compiler.register_ir(LogicalProgram, _verify_logical_source)
    compiler.register_ir(LogicalIR, verify_logical_ir)
    compiler.register_ir(NAProgram, verify_na_program)
    compiler.register_ir(ZonedPlan, verify_zoned_plan)
    compiler.register_pipeline(
        Pipeline(NA_PIPELINE, (parse_qasm, normalize_na, schedule_with_zap))
    )
    compiler.register_pipeline(
        Pipeline(
            LOGICAL_NA_PIPELINE,
            (freeze_logical, normalize_na, schedule_with_zap),
        )
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


def _package_sc_result(result: CompilationResult) -> CompilationResult:
    if type(result.output) not in (SCNativeProgram, _RotationNativeProgram):
        return result
    program, layout = to_sc_simulator_program(result.output)
    return ExecutableCompilationResult(
        output=result.output,
        route=result.route,
        program=program,
        resource_layout=layout,
    )


def _package_na_result(result: CompilationResult) -> CompilationResult:
    if type(result.output) is not ZonedPlan:
        return result
    program, layout = to_na_simulator_program(result.output)
    return ExecutableCompilationResult(
        output=result.output,
        route=result.route,
        program=program,
        resource_layout=layout,
    )


def compile_qasm_to_sc(
    source: str | QasmSource,
    backend: SCQubitSimulator,
    *,
    emit: str = SCNativeProgram.IR_ID,
    filename: str | None = None,
    seed: int = 0,
) -> CompilationResult:
    """Compile OpenQASM to an executable SC result at the final boundary."""

    return _package_sc_result(
        create_sc_pipeline().compile(
            _qasm_source(source, filename),
            pipeline=SC_PIPELINE,
            emit=emit,
            context=CompileContext(target=backend, options={"seed": seed}),
        )
    )


def compile_to_sc(
    source: LogicalProgram,
    backend: SCQubitSimulator,
    *,
    emit: str = SCNativeProgram.IR_ID,
    seed: int = 0,
) -> CompilationResult:
    """Compile a LogicalProgram to an executable superconducting result.

    Lowering snapshots the source without editing it. Bind symbolic parameters
    before compiling; measurements must be terminal, and classical conditions
    and direct physical controls are unsupported. Register views expand into
    scalar gates when frozen.

    Args:
        source: An exact LogicalProgram containing static numeric gates.
        backend: SCQubitSimulator supplying the capacity and coupling graph.
        emit: Representation to return; defaults to SCNativeProgram.IR_ID.
            Supported values are LogicalProgram.IR_ID, LogicalIR.IR_ID,
            SCProgram.IR_ID, and SCNativeProgram.IR_ID.
        seed: Routing seed, default 0.

    Returns:
        ExecutableCompilationResult at the final boundary; CompilationResult
        for earlier boundaries. Emitting LogicalProgram.IR_ID runs no passes:
        the LogicalProgram input is returned unchanged. Static gate and target
        validation begins at later boundaries.

    Raises:
        ValidationError: If the source type or an IR boundary is invalid.
        EmitNotFoundError: If emit is not a boundary of the selected route.
        PassError: If snapshotting or target lowering fails.
    """

    return _package_sc_result(
        create_sc_pipeline().compile(
            source,
            pipeline=LOGICAL_SC_PIPELINE,
            emit=emit,
            context=CompileContext(target=backend, options={"seed": seed}),
        )
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

    return _package_sc_result(
        _create_sc_rotation_pipeline().compile(
            _qasm_source(source, filename),
            pipeline=_SC_ROTATION_PIPELINE,
            emit=emit,
            context=CompileContext(target=backend, options={"seed": seed}),
        )
    )


def compile_qasm_to_na(
    source: str | QasmSource,
    architecture: Mapping[str, object],
    *,
    emit: str = ZonedPlan.IR_ID,
    filename: str | None = None,
) -> CompilationResult:
    """Compile OpenQASM to an executable ZAP-scheduled result at the final boundary."""

    return _package_na_result(
        create_na_pipeline().compile(
            _qasm_source(source, filename),
            pipeline=NA_PIPELINE,
            emit=emit,
            context=CompileContext(target=architecture),
        )
    )


def compile_to_na(
    source: LogicalProgram,
    architecture: Mapping[str, object],
    *,
    emit: str = ZonedPlan.IR_ID,
) -> CompilationResult:
    """Compile a LogicalProgram to an executable neutral-atom result.

    Lowering snapshots the source without editing it. Bind symbolic parameters
    before compiling; measurements must be terminal, and classical conditions
    and direct physical controls are unsupported. Register views expand into
    scalar gates when frozen.
    NA lowering rejects SX and Reset operations.

    Args:
        source: An exact LogicalProgram containing static numeric gates.
        architecture: ZAP architecture mapping, as returned by load_architecture.
        emit: Representation to return; defaults to ZonedPlan.IR_ID.
            Supported values are LogicalProgram.IR_ID, LogicalIR.IR_ID,
            NAProgram.IR_ID, and ZonedPlan.IR_ID.

    Returns:
        ExecutableCompilationResult at the final boundary; CompilationResult
        for earlier boundaries. Emitting LogicalProgram.IR_ID runs no passes:
        the LogicalProgram input is returned unchanged. Static gate and target
        validation begins at later boundaries.

    Raises:
        ValidationError: If the source type or an IR boundary is invalid.
        EmitNotFoundError: If emit is not a boundary of the selected route.
        PassError: If snapshotting or target lowering fails.
    """

    return _package_na_result(
        create_na_pipeline().compile(
            source,
            pipeline=LOGICAL_NA_PIPELINE,
            emit=emit,
            context=CompileContext(target=architecture),
        )
    )
