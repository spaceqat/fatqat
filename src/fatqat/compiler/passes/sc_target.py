"""Route unified SC programs and lower events into concrete native bases."""

from __future__ import annotations

import math
from collections.abc import Callable

from ... import operations as ops
from ..._sc_target import SCTarget
from ...operations.fixed_gates import CZGate, SwapGate
from ...operations.parametric_gates import RX, RZ
from ...registers import ClassicalRegister
from ...simulator import SCQubitSimulator
from ...simulator.fake_superconducting import _SCQubitRotationSimulator
from ..algorithms import RouteSwap, SabreResult, SiteId, sabre_map
from ..core import CompileContext
from ..dialects.sc_gate import MeasureOp, SCNode, SCProgram
from ..dialects.sc_native import (
    NativeGate,
    NativeInstruction,
    NativeMeasure,
    NativeReset,
    SCNativeProgram,
    _RotationNativeProgram,
    _verify_rotation_native_program,
    verify_sc_native_program,
)
from ..errors import ValidationError


def lower_sc_to_native_program(
    source: SCProgram,
    backend: SCTarget | SCQubitSimulator,
    *,
    seed: int = 0,
) -> SCNativeProgram:
    """Route and lower one SC program to the canonical native basis.

    An SCTarget supplies only capacity and undirected CZ edges. An
    SCQubitSimulator routes on its construction target, limited to the CZ
    operands its implementation map names, and every emitted gate is still
    checked against that map's ordered operands.
    """

    if isinstance(backend, SCTarget):
        constraints = backend
    elif isinstance(backend, SCQubitSimulator):
        constraints = _legacy_constraints(backend)
    else:
        raise TypeError("SC lowering requires SCTarget or SCQubitSimulator")
    routed = sabre_map(
        source,
        tuple(range(constraints.num_qubits)),
        frozenset(constraints.couplings),
        seed=seed,
    )
    operations = _lower_events(source, routed, _lower_sx_node, _sx_swap)
    native = SCNativeProgram(
        operations,
        routed.initial_layout,
        routed.final_layout,
        classical_registers=_declared_classical_registers(source),
    )
    verify_sc_native_program(native)
    if isinstance(backend, SCTarget):
        _verify_against_target(native, backend)
    else:
        _verify_against_backend(native.operations, native.initial_layout, backend)
    return native


def _legacy_constraints(backend: SCQubitSimulator) -> SCTarget:
    # A uniformly available CZ names no operands and must not widen routing.
    target = backend.get_compiler_target()
    named = {
        frozenset(sites)
        for sites in backend.implementation_map.device_operands_for(ops.CZ)
    }
    return SCTarget(
        num_qubits=target.num_qubits,
        couplings=(edge for edge in target.couplings if frozenset(edge) in named),
    )


def _lower_sc_to_rotation_program(
    source: SCProgram,
    backend: _SCQubitRotationSimulator,
    *,
    seed: int = 0,
) -> _RotationNativeProgram:
    """Route and lower one SC program to the private rotation basis."""

    if not isinstance(backend, _SCQubitRotationSimulator):
        raise TypeError("rotation lowering requires _SCQubitRotationSimulator")
    routed = _route(source, backend, seed)
    operations = _lower_events(source, routed, _lower_rotation_node, _rotation_swap)
    target = _RotationNativeProgram(
        operations,
        routed.initial_layout,
        routed.final_layout,
        classical_registers=_declared_classical_registers(source),
    )
    _verify_rotation_native_program(target)
    _verify_against_backend(target.operations, target.initial_layout, backend)
    return target


def _declared_classical_registers(source: SCProgram) -> tuple[ClassicalRegister, ...]:
    registers: list[ClassicalRegister] = []
    seen: set[int] = set()
    for clbit in source.clbits:
        register = clbit.register
        if id(register) not in seen:
            seen.add(id(register))
            registers.append(register)
    return tuple(registers)


def _route(source: SCProgram, backend, seed: int) -> SabreResult:
    implementation_map = backend.implementation_map
    couplings = implementation_map.device_operands_for(ops.CZ)
    return sabre_map(source, backend.device_sites, couplings, seed=seed)


NodeLowerer = Callable[[SCNode, tuple[SiteId, ...]], tuple[tuple[object, tuple], ...]]
SwapLowerer = Callable[
    [tuple[SiteId, SiteId]], tuple[tuple[object, tuple[SiteId, ...]], ...]
]


def _lower_events(
    source: SCProgram,
    routed: SabreResult,
    lower_node: NodeLowerer,
    lower_swap: SwapLowerer,
) -> tuple[NativeInstruction, ...]:
    operations: list[NativeInstruction] = []
    deferred_measurements: list[tuple[int, SCNode]] = []

    def operation_id() -> str:
        return f"native.{len(operations)}"

    for event in routed.events:
        if isinstance(event, RouteSwap):
            for operation, sites in lower_swap(event.sites):
                operations.append(
                    NativeGate(
                        operation_id(),
                        operation,
                        sites,
                        (),
                        event.swap_id,
                    )
                )
            continue

        node = source.nodes[event.node_id]
        if type(node.instruction) is MeasureOp:
            deferred_measurements.append((event.node_id, node))
            continue
        if type(node.instruction) is type(ops.Reset):
            operations.append(
                NativeReset(operation_id(), event.sites[0], node.origin_ids)
            )
            continue
        for operation, sites in lower_node(node, event.sites):
            operations.append(
                NativeGate(
                    operation_id(),
                    operation,
                    sites,
                    node.origin_ids,
                )
            )

    final_layout = dict(routed.final_layout)
    for _node_id, node in sorted(deferred_measurements):
        operations.append(
            NativeMeasure(
                operation_id(),
                final_layout[node.qubits[0]],
                node.clbits[0],
                node.origin_ids,
            )
        )
    return tuple(operations)


def _lower_sx_node(
    node: SCNode, sites: tuple[SiteId, ...]
) -> tuple[tuple[object, tuple], ...]:
    instruction = node.instruction
    if type(instruction) is RX:
        theta = instruction.theta
        site = (sites[0],)
        return (
            (ops.RZ(math.pi / 2), site),
            (ops.SX, site),
            (ops.RZ(theta + math.pi), site),
            (ops.SX, site),
            (ops.RZ(math.pi / 2), site),
        )
    if type(instruction) in (RZ, CZGate):
        return ((instruction, sites),)
    if type(instruction) is SwapGate:
        return _sx_swap((sites[0], sites[1]))
    raise ValidationError(f"cannot lower SC instruction {type(instruction).__name__}")


def _lower_rotation_node(
    node: SCNode, sites: tuple[SiteId, ...]
) -> tuple[tuple[object, tuple], ...]:
    instruction = node.instruction
    if type(instruction) in (RX, RZ, CZGate):
        return ((instruction, sites),)
    if type(instruction) is SwapGate:
        return _rotation_swap((sites[0], sites[1]))
    raise ValidationError(f"cannot lower SC instruction {type(instruction).__name__}")


def _sx_h(site: SiteId) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    target = (site,)
    return (
        (ops.RZ(math.pi / 2), target),
        (ops.SX, target),
        (ops.RZ(math.pi / 2), target),
    )


def _rotation_h(site: SiteId) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    target = (site,)
    return (
        (ops.RZ(math.pi / 2), target),
        (ops.RX(math.pi / 2), target),
        (ops.RZ(math.pi / 2), target),
    )


def _cx(
    control: SiteId,
    target: SiteId,
    lower_h: Callable[[SiteId], tuple[tuple[object, tuple[SiteId, ...]], ...]],
) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    return lower_h(target) + ((ops.CZ, (control, target)),) + lower_h(target)


def _sx_swap(
    sites: tuple[SiteId, SiteId],
) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    first, second = sites
    return (
        _cx(first, second, _sx_h)
        + _cx(second, first, _sx_h)
        + _cx(first, second, _sx_h)
    )


def _rotation_swap(
    sites: tuple[SiteId, SiteId],
) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    first, second = sites
    return (
        _cx(first, second, _rotation_h)
        + _cx(second, first, _rotation_h)
        + _cx(first, second, _rotation_h)
    )


def _verify_against_backend(operations, initial_layout, backend) -> None:
    legal_sites = frozenset(backend.device_sites)
    if any(site not in legal_sites for _, site in initial_layout):
        raise ValidationError("native layout names a site outside backend")
    implementation_map = backend.implementation_map
    for instruction in operations:
        if isinstance(instruction, NativeGate):
            sites = instruction.sites
            rule = implementation_map.implementation_for(
                instruction.operation,
                device_operands=sites,
            )
            if rule is None:
                raise ValidationError(
                    f"native operation {instruction.operation.name} is illegal "
                    f"on sites {sites}"
                )
        elif instruction.site not in legal_sites:
            raise ValidationError("native instruction names a site outside backend")


def _verify_against_target(native: SCNativeProgram, target: SCTarget) -> None:
    legal_sites = frozenset(range(target.num_qubits))
    edges = frozenset(frozenset(edge) for edge in target.couplings)
    for layout in (native.initial_layout, native.final_layout):
        if any(site not in legal_sites for _, site in layout):
            raise ValidationError("native layout names a site outside target")
    for instruction in native.operations:
        is_gate = isinstance(instruction, NativeGate)
        sites = instruction.sites if is_gate else (instruction.site,)
        if any(site not in legal_sites for site in sites):
            raise ValidationError("native instruction names a site outside target")
        if (
            is_gate
            and type(instruction.operation) is CZGate
            and (frozenset(sites) not in edges)
        ):
            raise ValidationError(
                f"native CZ on sites {sites} is outside target couplings"
            )


class LowerScToNativePass:
    name = "lower-sc-to-native"
    source_type = SCProgram
    target_type = SCNativeProgram

    def run(self, source: SCProgram, context: CompileContext) -> SCNativeProgram:
        return lower_sc_to_native_program(
            source,
            context.target,
            seed=context.options.get("seed", 0),
        )


class _LowerScToRotationPass:
    name = "lower-sc-to-rotation"
    source_type = SCProgram
    target_type = _RotationNativeProgram

    def run(self, source: SCProgram, context: CompileContext) -> _RotationNativeProgram:
        return _lower_sc_to_rotation_program(
            source,
            context.target,
            seed=context.options.get("seed", 0),
        )


lower_sc_to_native = LowerScToNativePass()
_lower_sc_to_rotation = _LowerScToRotationPass()
