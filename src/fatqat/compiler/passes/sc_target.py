"""Route unified SC programs and lower events into concrete native bases."""

from __future__ import annotations

import math
from collections.abc import Callable

from ... import operations as ops
from ...operations.fixed_gates import CZGate, SwapGate
from ...operations.parametric_gates import RX, RZ
from ...operations.reset import ResetGate
from ...simulator import SCQubitGoogleSimulator, SCQubitIBMSimulator
from ..algorithms import RouteSwap, SabreResult, SiteId, sabre_map
from ..core import CompileContext
from ..dialects.sc_gate import MeasureOp, SCNode, SCProgram
from ..dialects.sc_native import (
    GoogleProgram,
    IBMProgram,
    NativeGate,
    NativeInstruction,
    NativeMeasure,
    NativeReset,
    verify_google_program,
    verify_ibm_program,
)
from ..errors import ValidationError


def lower_sc_to_ibm_program(
    source: SCProgram,
    backend: SCQubitIBMSimulator,
    *,
    seed: int = 0,
) -> IBMProgram:
    """Route and lower one SC program to the IBM-style native basis."""

    if not isinstance(backend, SCQubitIBMSimulator):
        raise TypeError("IBM lowering requires SCQubitIBMSimulator")
    routed = _route(source, backend, seed)
    operations = _lower_events(source, routed, _lower_ibm_node, _ibm_swap)
    target = IBMProgram(operations, routed.initial_layout, routed.final_layout)
    verify_ibm_program(target)
    _verify_against_backend(target.operations, target.initial_layout, backend)
    return target


def lower_sc_to_google_program(
    source: SCProgram,
    backend: SCQubitGoogleSimulator,
    *,
    seed: int = 0,
) -> GoogleProgram:
    """Route and lower one SC program to the Google-style native basis."""

    if not isinstance(backend, SCQubitGoogleSimulator):
        raise TypeError("Google lowering requires SCQubitGoogleSimulator")
    routed = _route(source, backend, seed)
    operations = _lower_events(source, routed, _lower_google_node, _google_swap)
    target = GoogleProgram(operations, routed.initial_layout, routed.final_layout)
    verify_google_program(target)
    _verify_against_backend(target.operations, target.initial_layout, backend)
    return target


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
        if type(node.instruction) is ResetGate:
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


def _lower_ibm_node(
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
        return _ibm_swap((sites[0], sites[1]))
    raise ValidationError(f"cannot lower SC instruction {type(instruction).__name__}")


def _lower_google_node(
    node: SCNode, sites: tuple[SiteId, ...]
) -> tuple[tuple[object, tuple], ...]:
    instruction = node.instruction
    if type(instruction) in (RX, RZ, CZGate):
        return ((instruction, sites),)
    if type(instruction) is SwapGate:
        return _google_swap((sites[0], sites[1]))
    raise ValidationError(f"cannot lower SC instruction {type(instruction).__name__}")


def _ibm_h(site: SiteId) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    target = (site,)
    return (
        (ops.RZ(math.pi / 2), target),
        (ops.SX, target),
        (ops.RZ(math.pi / 2), target),
    )


def _google_h(site: SiteId) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
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


def _ibm_swap(
    sites: tuple[SiteId, SiteId],
) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    first, second = sites
    return (
        _cx(first, second, _ibm_h)
        + _cx(second, first, _ibm_h)
        + _cx(first, second, _ibm_h)
    )


def _google_swap(
    sites: tuple[SiteId, SiteId],
) -> tuple[tuple[object, tuple[SiteId, ...]], ...]:
    first, second = sites
    return (
        _cx(first, second, _google_h)
        + _cx(second, first, _google_h)
        + _cx(first, second, _google_h)
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


class LowerScToIbmPass:
    name = "lower-sc-to-ibm"
    source_type = SCProgram
    target_type = IBMProgram

    def run(self, source: SCProgram, context: CompileContext) -> IBMProgram:
        return lower_sc_to_ibm_program(
            source,
            context.target,
            seed=context.options.get("seed", 0),
        )


class LowerScToGooglePass:
    name = "lower-sc-to-google"
    source_type = SCProgram
    target_type = GoogleProgram

    def run(self, source: SCProgram, context: CompileContext) -> GoogleProgram:
        return lower_sc_to_google_program(
            source,
            context.target,
            seed=context.options.get("seed", 0),
        )


lower_sc_to_ibm = LowerScToIbmPass()
lower_sc_to_google = LowerScToGooglePass()
