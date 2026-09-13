"""Physical-site native compiler IR for superconducting targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

from ...operations import Operation
from ...operations.fixed_gates import CZGate, SXGate, XGate, iSwapGate
from ...operations.parametric_gates import RX, RY, RZ
from ...registers import ClassicalRegister, RegisterRef
from ..algorithms.sabre import LayoutSnapshot, SabreResult, SiteId
from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class NativeGate:
    operation_id: str
    operation: Operation
    sites: tuple[SiteId, ...]
    origin_ids: tuple[str, ...]
    generated_by: str | None = None


@dataclass(frozen=True, slots=True)
class NativeMeasure:
    operation_id: str
    site: SiteId
    clbit: RegisterRef
    origin_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeReset:
    operation_id: str
    site: SiteId
    origin_ids: tuple[str, ...]


NativeInstruction: TypeAlias = NativeGate | NativeMeasure | NativeReset


@dataclass(frozen=True, slots=True)
class SCNativeProgram:
    """Canonical physical-site program for the public SC target.

    Args:
        operations: Native gates, measurements, and resets in execution order.
        initial_layout: Logical qubit to physical site pairs before routing.
        final_layout: Logical qubit to physical site pairs after routing.
        classical_registers: Keyword-only tuple of original classical register
            objects in output declaration order, including unused registers.
            Each register contributes all slots in increasing index order.
            The default ``None`` lets the simulator bridge infer registers
            from first measurement occurrence for legacy native programs;
            that fallback cannot recover unused registers or source order.
            ``()`` explicitly declares no classical registers. Compiler
            lowering always supplies a tuple.

    The native verifier and simulator bridge validate supplied declarations:
    register objects must be distinct ClassicalRegister instances, and every
    measurement destination must belong to a declared register. Names need
    not be unique. Construction itself does not perform this validation.
    """

    IR_ID: ClassVar[str] = "sc.native.v1"

    operations: tuple[NativeInstruction, ...]
    initial_layout: LayoutSnapshot
    final_layout: LayoutSnapshot
    classical_registers: tuple[ClassicalRegister, ...] | None = field(
        default=None, kw_only=True
    )


@dataclass(frozen=True, slots=True)
class _RotationNativeProgram:
    """Physical-site program for the private rotation target."""

    IR_ID: ClassVar[str] = "sc.rotation.native.v1"

    operations: tuple[NativeInstruction, ...]
    initial_layout: LayoutSnapshot
    final_layout: LayoutSnapshot
    classical_registers: tuple[ClassicalRegister, ...] | None = field(
        default=None, kw_only=True
    )


def verify_sc_native_program(program: object) -> None:
    _verify_native_program(
        program,
        SCNativeProgram,
        (XGate, SXGate, RZ, CZGate),
        "SC",
    )


def _verify_rotation_native_program(program: object) -> None:
    _verify_native_program(
        program,
        _RotationNativeProgram,
        (RX, RY, RZ, iSwapGate, CZGate),
        "rotation",
    )


def _verify_native_program(
    program: object,
    program_type: type[SCNativeProgram] | type[_RotationNativeProgram],
    gate_types: tuple[type[Operation], ...],
    label: str,
) -> None:
    if type(program) is not program_type:
        raise ValidationError(f"expected {program_type.__name__}")
    if not isinstance(program.operations, tuple):
        raise ValidationError(f"{label} operations must be a tuple")
    try:
        SabreResult((), program.initial_layout, program.final_layout)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc

    operation_ids: set[str] = set()
    for instruction in program.operations:
        if type(instruction) not in (NativeGate, NativeMeasure, NativeReset):
            raise ValidationError(f"unsupported {label} native instruction")
        _verify_operation_id(instruction.operation_id, operation_ids)

        if type(instruction) is NativeGate:
            if type(instruction.operation) not in gate_types:
                raise ValidationError(
                    f"unsupported {label} native operation: "
                    f"{type(instruction.operation).__name__}"
                )
            if len(instruction.sites) != instruction.operation.num_subsystems:
                raise ValidationError("native gate operand count is invalid")
            _verify_sites(instruction.sites)
            semantic = bool(instruction.origin_ids) and instruction.generated_by is None
            routed = not instruction.origin_ids and bool(instruction.generated_by)
            if not (semantic or routed):
                raise ValidationError(
                    "native gate must have semantic origins or route provenance"
                )
            _verify_origins(instruction.origin_ids, allow_empty=routed)
            continue

        _verify_site(instruction.site)
        _verify_origins(instruction.origin_ids, allow_empty=False)
        if type(instruction) is NativeMeasure and (
            type(instruction.clbit) is not RegisterRef
            or not isinstance(instruction.clbit.register, ClassicalRegister)
        ):
            raise ValidationError("native measurement output must be a clbit")

    _verify_native_classical_registers(program)


def _verify_native_classical_registers(
    program: SCNativeProgram | _RotationNativeProgram,
) -> None:
    """Validate explicit output declarations without changing legacy inference."""
    registers = program.classical_registers
    if registers is None:
        return
    if not isinstance(registers, tuple):
        raise ValidationError("native classical registers must be a tuple")
    seen: set[int] = set()
    for register in registers:
        if not isinstance(register, ClassicalRegister):
            raise ValidationError(
                "native classical registers must be ClassicalRegister instances"
            )
        if id(register) in seen:
            raise ValidationError("duplicate native classical register")
        seen.add(id(register))

    for instruction in program.operations:
        if not isinstance(instruction, NativeMeasure):
            continue
        if type(instruction.clbit) is not RegisterRef or not isinstance(
            instruction.clbit.register, ClassicalRegister
        ):
            raise ValidationError("native measurement output must be a clbit")
        if id(instruction.clbit.register) not in seen:
            raise ValidationError(
                "native measurement references an undeclared classical register"
            )


def _verify_operation_id(operation_id: str, seen: set[str]) -> None:
    if not isinstance(operation_id, str) or not operation_id:
        raise ValidationError("native operation ID must be a non-empty string")
    if operation_id in seen:
        raise ValidationError(f"duplicate native operation ID: {operation_id}")
    seen.add(operation_id)


def _verify_origins(origin_ids: tuple[str, ...], *, allow_empty: bool) -> None:
    if not isinstance(origin_ids, tuple):
        raise ValidationError("native origin IDs must be a tuple")
    if not allow_empty and not origin_ids:
        raise ValidationError("native origin IDs must not be empty")
    if any(not isinstance(item, str) or not item for item in origin_ids):
        raise ValidationError("native origin IDs must contain non-empty strings")


def _verify_sites(sites: tuple[SiteId, ...]) -> None:
    if not isinstance(sites, tuple) or not sites:
        raise ValidationError("native gate sites must be a non-empty tuple")
    for site in sites:
        _verify_site(site)
    if len(set(sites)) != len(sites):
        raise ValidationError("native gate sites must be distinct")


def _verify_site(site: SiteId) -> None:
    if isinstance(site, bool) or not isinstance(site, (int, str)):
        raise ValidationError("native site must be an integer or string")
    if isinstance(site, str) and not site:
        raise ValidationError("native site string must not be empty")
