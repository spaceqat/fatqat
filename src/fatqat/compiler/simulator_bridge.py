"""Convert hardware compiler IR into the simulator's public Program API."""

from __future__ import annotations

from .. import operations as ops
from ..operations.fixed_gates import CZGate
from ..program import Program
from ..registers import ClassicalRegister, QuantumRegister
from ..resource_layout import ResourceLayout
from .dialects.na_zoned import GateBatch, ZonedPlan, verify_zoned_plan
from .dialects.sc_native import (
    GoogleProgram,
    IBMProgram,
    NativeGate,
    NativeMeasure,
    NativeReset,
)
from .errors import ValidationError


def to_sc_simulator_program(
    native: IBMProgram | GoogleProgram,
) -> tuple[Program, ResourceLayout]:
    """Project an SC native program onto fixed simulator site references."""

    if type(native) not in (IBMProgram, GoogleProgram):
        raise TypeError("native must be IBMProgram or GoogleProgram")

    sites: list[int | str] = []

    def remember(site: int | str) -> None:
        if site not in sites:
            sites.append(site)

    for layout in (native.initial_layout, native.final_layout):
        for _logical, site in layout:
            remember(site)
    classical_registers: list[ClassicalRegister] = []
    for instruction in native.operations:
        if isinstance(instruction, NativeGate):
            for site in instruction.sites:
                remember(site)
        else:
            remember(instruction.site)
        if isinstance(instruction, NativeMeasure):
            register = instruction.clbit.register
            if all(register is not existing for existing in classical_registers):
                classical_registers.append(register)

    physical = QuantumRegister(len(sites), name="physical")
    program = Program([physical], classical_registers)
    refs = {site: physical[index] for index, site in enumerate(sites)}

    for instruction in native.operations:
        if isinstance(instruction, NativeGate):
            targets = tuple(refs[site] for site in instruction.sites)
            program.add(instruction.operation, targets)
        elif isinstance(instruction, NativeReset):
            program.add(ops.Reset, refs[instruction.site])
        else:
            program.measure(refs[instruction.site], instruction.clbit)

    return program, ResourceLayout({ref: site for site, ref in refs.items()})


def to_na_simulator_program(plan: ZonedPlan) -> tuple[Program, ResourceLayout]:
    """Project a verified neutral-atom physical plan into a simulator program."""

    verify_zoned_plan(plan)

    atom_refs = set(plan.atoms)
    clbit_refs = set(plan.clbits)
    quantum_registers: list[QuantumRegister] = []
    seen_quantum_registers: set[QuantumRegister] = set()
    for atom in plan.atoms:
        if atom.register in seen_quantum_registers:
            continue
        if any(
            atom.register[index] not in atom_refs for index in range(atom.register.size)
        ):
            raise ValidationError(
                "ZonedPlan atoms must include each ref of each referenced quantum register"
            )
        seen_quantum_registers.add(atom.register)
        quantum_registers.append(atom.register)

    classical_registers: list[ClassicalRegister] = []
    seen_classical_registers: set[ClassicalRegister] = set()
    for clbit in plan.clbits:
        if clbit.register in seen_classical_registers:
            continue
        if any(
            clbit.register[index] not in clbit_refs
            for index in range(clbit.register.size)
        ):
            raise ValidationError(
                "ZonedPlan clbits must include each ref of each referenced classical register"
            )
        seen_classical_registers.add(clbit.register)
        classical_registers.append(clbit.register)

    program = Program(quantum_registers, classical_registers)
    if plan.atoms:
        program.add(ops.Put, plan.atoms)
    for event in plan.events:
        if type(event) is not GateBatch:
            continue
        cz_gates = [gate for gate in event.gates if type(gate.operation) is CZGate]
        for gate in cz_gates:
            program.add(ops.Pair, gate.atoms)
        for gate in event.gates:
            program.add(gate.operation, gate.atoms)
        for gate in cz_gates:
            program.add(ops.Unpair, gate.atoms)

    for measurement in plan.terminal_measurements:
        program.measure(measurement.atom, measurement.clbit)

    return program, ResourceLayout(
        {atom: index for index, atom in enumerate(plan.atoms)}
    )
