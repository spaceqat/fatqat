"""Construct and run the compiler's explicit translation pipelines."""

from __future__ import annotations

from collections.abc import Mapping

from .._sc_target import SCTarget
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
    """Create a compiler with the built-in superconducting pipelines.

    Choose a pipeline by its source type:

    - `SC_PIPELINE` (``"qasm-to-sc"``): accepts `QasmSource`.
    - `LOGICAL_SC_PIPELINE` (``"logical-to-sc"``): accepts `LogicalProgram`.

    To lower to native gates, set ``CompileContext.target`` to an `SCTarget`
    or `SCQubitSimulator`. The optional ``options["seed"]`` sets the routing
    seed (default ``0``).

    Returns:
        A new `Compiler` with both pipelines registered.
    """

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
    """Create a compiler with the built-in neutral-atom pipelines.

    Choose a pipeline by its source type:

    - `NA_PIPELINE` (``"qasm-to-na-zap"``): accepts `QasmSource`.
    - `LOGICAL_NA_PIPELINE` (``"logical-to-na"``): accepts `LogicalProgram`.

    To produce a scheduled plan, pass an architecture such as one from
    `fatqat.compiler.algorithms.load_architecture` as ``CompileContext.target``.

    Returns:
        A new `Compiler` with both pipelines registered.
    """

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
    backend: SCTarget | SCQubitSimulator,
    *,
    emit: str = SCNativeProgram.IR_ID,
    filename: str | None = None,
    seed: int = 0,
) -> CompilationResult:
    """Compile OpenQASM for a superconducting target.

    Args:
        source: OpenQASM 2 or 3 text, or an existing QasmSource.
        backend: SCTarget or SCQubitSimulator that supplies capacity and
            connectivity. The gate rules of an SCQubitSimulator also apply.
        emit: Stop at one of these representations:

            - ``"qasm.text.v1"`` (`QasmSource.IR_ID`): source text and label
              before parsing. No passes run.
            - ``"gate.logical.v1"`` (`LogicalIR.IR_ID`): device-independent
              gate representation.
            - ``"sc.gate.v1"`` (`SCProgram.IR_ID`): normalized superconducting
              gates before routing and lowering to native gates.
            - ``"sc.native.v1"`` (`SCNativeProgram.IR_ID`, default): gates
              on physical sites, with a simulator-ready result.
        filename: Optional source label retained on the QasmSource boundary.
            It cannot be supplied when source is already a QasmSource.
        seed: Routing seed, default 0.

    Returns:
        An `ExecutableCompilationResult` at the final native stage, or a
            `CompilationResult` for inspection at an earlier stage.

    Raises:
        TypeError: If source is neither text nor an exact QasmSource.
        ValueError: If filename accompanies an existing QasmSource.
        ValidationError: If an IR boundary is invalid.
        EmitNotFoundError: If emit is not a boundary of the SC route.
        PassError: If parsing, normalization, routing, or lowering fails.
    """

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
    backend: SCTarget | SCQubitSimulator,
    *,
    emit: str = SCNativeProgram.IR_ID,
    seed: int = 0,
) -> CompilationResult:
    """Compile a LogicalProgram to an executable superconducting result.

    Lowering snapshots the source without editing it. The SC route decomposes
    Toffoli (CCX) gates before routing. Bind symbolic parameters before compiling;
    measurements must be terminal, and classical conditions and direct physical
    controls are unsupported. Register views expand into scalar gates when frozen.

    Args:
        source: An exact LogicalProgram containing static numeric gates.
        backend: SCTarget or SCQubitSimulator that supplies the capacity and
            coupling graph. The gate rules of an SCQubitSimulator also apply.
        emit: Stop at one of these representations:

            - ``"gate.logical.source.v1"`` (`LogicalProgram.IR_ID`): return
              the input program unchanged. No passes run.
            - ``"gate.logical.v1"`` (`LogicalIR.IR_ID`): device-independent
              gate representation.
            - ``"sc.gate.v1"`` (`SCProgram.IR_ID`): normalized superconducting
              gates before routing and lowering to native gates.
            - ``"sc.native.v1"`` (`SCNativeProgram.IR_ID`, default): gates
              on physical sites, with a simulator-ready result.
        seed: Routing seed, default 0.

    Returns:
        An `ExecutableCompilationResult` at the final stage, or a
            `CompilationResult` for inspection at an earlier stage. If ``emit`` selects the input ID, the input
            program is returned unchanged. Static gate and target validation
            begins at later stages.

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
    """Compile OpenQASM for a neutral-atom architecture.

    Args:
        source: OpenQASM 2 or 3 text, or an existing QasmSource.
        architecture: ZAP architecture mapping, as returned by
            load_architecture.
        emit: Stop at one of these representations:

            - ``"qasm.text.v1"`` (`QasmSource.IR_ID`): source text and label
              before parsing. No passes run.
            - ``"gate.logical.v1"`` (`LogicalIR.IR_ID`): device-independent
              gate representation.
            - ``"na.gate.v1"`` (`NAProgram.IR_ID`): neutral-atom gate
              representation before physical scheduling.
            - ``"na.zoned.plan.v1"`` (`ZonedPlan.IR_ID`, default): scheduled
              device operations, with a simulator-ready result.
        filename: Optional source label retained on the QasmSource boundary.
            It cannot be supplied when source is already a QasmSource.

    Returns:
        An `ExecutableCompilationResult` at the final zoned-plan stage, or a
            `CompilationResult` for inspection at an earlier stage.

    Raises:
        TypeError: If source is neither text nor an exact QasmSource.
        ValueError: If filename accompanies an existing QasmSource.
        ValidationError: If an IR boundary is invalid.
        EmitNotFoundError: If emit is not a boundary of the NA route.
        PassError: If parsing, normalization, or ZAP scheduling fails.
    """

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
        emit: Stop at one of these representations:

            - ``"gate.logical.source.v1"`` (`LogicalProgram.IR_ID`): return
              the input program unchanged. No passes run.
            - ``"gate.logical.v1"`` (`LogicalIR.IR_ID`): device-independent
              gate representation.
            - ``"na.gate.v1"`` (`NAProgram.IR_ID`): neutral-atom gate
              representation before physical scheduling.
            - ``"na.zoned.plan.v1"`` (`ZonedPlan.IR_ID`, default): scheduled
              device operations, with a simulator-ready result.

    Returns:
        An `ExecutableCompilationResult` at the final stage, or a
            `CompilationResult` for inspection at an earlier stage. If ``emit`` selects the input ID, the input
            program is returned unchanged. Static gate and target validation
            begins at later stages.

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
