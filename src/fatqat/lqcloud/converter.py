"""Prepare the cloud executable at the compiler boundary, before submission."""

from dataclasses import dataclass
from typing import Any

from .. import operations as ops
from ..compiler.core import CompilationResult
from ..compiler.dialects.lq_native import LQNativeProgram, verify_lq_native_program
from ..compiler.dialects.sc_native import NativeGate, NativeReset
from ..compiler.targets import _SCTarget


def _load_sdk():
    try:
        import lqcloud
    except ImportError as exc:
        raise ImportError(
            "LQCloud requires Python 3.12 and the optional SDK: "
            "pip install 'fatqat[lqcloud]'"
        ) from exc
    return lqcloud


@dataclass(frozen=True, slots=True)
class LQCloudCompilationResult(CompilationResult[LQNativeProgram]):
    """Compiled native IR and a ready-to-submit SDK circuit.

    Use ``backend.run(compiled)`` to execute. ``output`` and ``route`` expose
    compiler IR and pass names. Treat ``program`` (an SDK QuantumCircuit) as
    read-only; ``physical_layout`` maps its compact wires to physical sites,
    not original logical qubits. ``target`` is the compilation snapshot and
    ``classical_dims`` follows the source's declared classical slot order.
    """

    program: Any
    physical_layout: tuple[int, ...]
    target: _SCTarget
    classical_dims: tuple[int, ...]


def _prepare_lq_result(
    result: CompilationResult, target: _SCTarget
) -> LQCloudCompilationResult:
    native = result.output
    verify_lq_native_program(native)
    if native.classical_registers is None:
        raise ValueError("cloud compilation requires declared classical registers")
    clbits = tuple(ref for reg in native.classical_registers for ref in reg)
    classical_indices = {ref: index for index, ref in enumerate(clbits)}
    sites = dict.fromkeys(site for _, site in native.initial_layout)
    for instruction in native.operations:
        sites.update(
            dict.fromkeys(
                instruction.sites
                if isinstance(instruction, NativeGate)
                else (instruction.site,)
            )
        )
    physical_layout = tuple(sites)
    indices = {site: index for index, site in enumerate(physical_layout)}
    sdk = _load_sdk()
    registers = [sdk.QuantumRegister(len(sites))] if sites else []
    if clbits:
        registers.append(sdk.ClassicalRegister(len(clbits)))
    circuit = sdk.QuantumCircuit(*registers)
    measurement_started = False
    for instruction in native.operations:
        if isinstance(instruction, NativeGate):
            operands = tuple(indices[site] for site in instruction.sites)
            if instruction.operation is ops.H:
                circuit.h(*operands)
            elif instruction.operation is ops.CZ:
                circuit.cz(*operands)
            else:
                circuit.rz(float(instruction.operation.theta), *operands)
        elif isinstance(instruction, NativeReset):
            circuit.reset(indices[instruction.site])
        else:
            if not measurement_started:
                circuit.barrier()
                measurement_started = True
            circuit.measure(
                indices[instruction.site], classical_indices[instruction.clbit]
            )
    return LQCloudCompilationResult(
        output=native,
        route=result.route,
        program=circuit,
        physical_layout=physical_layout,
        target=target,
        classical_dims=(2,) * len(clbits),
    )
