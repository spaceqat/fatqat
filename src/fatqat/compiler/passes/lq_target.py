"""Lower routed SC operations to LogicalQubit's H/RZ/CZ native subset."""

from ... import operations as ops
from ...operations.fixed_gates import CZGate, SwapGate
from ...operations.parametric_gates import RX, RZ
from ..core import CompileContext
from ..dialects.lq_native import LQNativeProgram, verify_lq_native_program
from ..dialects.sc_gate import SCProgram
from ..dialects.sc_native import NativeGate
from ..errors import ValidationError
from ..targets import _SCTarget
from .sc_target import _declared_classical_registers, _lower_events, _route, _cx


def lower_sc_to_lq_program(
    source: SCProgram, target: _SCTarget, *, seed: int = 0
) -> LQNativeProgram:
    """Route and lower SC gates using a LogicalQubit device snapshot."""
    if target.profile != "lqcloud":
        raise TypeError("LQ lowering requires a lqcloud target")
    if any(ref.register.dim != 2 for ref in source.qubits + source.clbits):
        raise ValidationError("LQ devices require qubit and binary classical registers")
    routed = _route(source, target, seed)
    result = LQNativeProgram(
        _lower_events(source, routed, _lower_node, _swap),
        routed.initial_layout,
        routed.final_layout,
        classical_registers=_declared_classical_registers(source),
    )
    verify_lq_native_program(result)
    legal_edges = {frozenset(edge) for edge in target.couplings}
    for instruction in result.operations:
        if isinstance(instruction, NativeGate):
            if not set(instruction.sites) <= set(target.sites):
                raise ValidationError("native instruction names a site outside target")
            if (
                instruction.operation is ops.CZ
                and frozenset(instruction.sites) not in legal_edges
            ):
                raise ValidationError("native CZ is outside target couplings")
        elif instruction.site not in target.sites:
            raise ValidationError("native instruction names a site outside target")
    return result


def _h(site):
    return ((ops.H, (site,)),)


def _swap(sites):
    first, second = sites
    return _cx(first, second, _h) + _cx(second, first, _h) + _cx(first, second, _h)


def _lower_node(node, sites):
    instruction = node.instruction
    if type(instruction) is RX:
        return _h(sites[0]) + ((ops.RZ(instruction.theta), sites),) + _h(sites[0])
    if type(instruction) in (RZ, CZGate):
        return ((instruction, sites),)
    if type(instruction) is SwapGate:
        return _swap(sites)
    raise ValidationError(f"cannot lower SC instruction {instruction.name}")


class LowerScToLQPass:
    name = "lower-sc-to-lqcloud"
    source_type = SCProgram
    target_type = LQNativeProgram

    def run(self, source: SCProgram, context: CompileContext) -> LQNativeProgram:
        return lower_sc_to_lq_program(
            source, context.target, seed=context.options.get("seed", 0)
        )


lower_sc_to_lq = LowerScToLQPass()
