"""Test-only explicit-coordinate atom assembly; never imported by production."""

from __future__ import annotations

from dataclasses import dataclass

from fatqat.emulator import AtomArrangement
from fatqat._index_allocation import _ClassicalAllocation, _EngineAllocation
from fatqat.emulator._core import planning
from fatqat.emulator._core.planning import _PulseLoweringContext
from fatqat.emulator.atom_3level.qutip_adapter import _Atom3LevelQutipAdapter
from fatqat.emulator.atom_3level.realization import (
    default_atom_3level_gate_implementation_map,
)
from fatqat.emulator.atom_3level.target import _Atom3LevelTarget
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.noise import LindbladImplementationMap, NoiseModel
from fatqat.program import AppliedOperation, Program
from fatqat.resource_layout import ResourceLayout


def _explicit_arrangement(coordinates):
    coordinates = tuple(tuple(point) for point in coordinates)
    arrangement = object.__new__(AtomArrangement)
    object.__setattr__(arrangement, "rows", 1)
    object.__setattr__(arrangement, "cols", len(coordinates))
    object.__setattr__(arrangement, "spacing", 1.0)
    object.__setattr__(arrangement, "coordinates", coordinates)
    return arrangement


@dataclass(frozen=True)
class ExplicitAtom3LevelAssembly:
    """Private test seam joining target lowering, scheduling, and QuTiP."""

    target: _Atom3LevelTarget
    engine_allocation: _EngineAllocation
    implementation_map: object
    adapter: _Atom3LevelQutipAdapter
    program: Program
    context: _PulseLoweringContext

    def lower(self, operation, target_sites):
        step = AppliedOperation(
            operation,
            tuple(self.program.quantum_registers[0][site] for site in target_sites),
        )
        return planning._lower_gate(
            step,
            target=self.target,
            context=self.context,
            gate_implementation_map=self.implementation_map,
            noise_model=NoiseModel(),
            lindblad_implementation_map=LindbladImplementationMap(),
        )

    @staticmethod
    def schedule(blocks, *, boundary_time=0.0):
        return schedule_pulse_run(blocks, boundary_time=boundary_time)

    def propagator(self, blocks, *, apply_final_frame=True):
        return self.adapter.propagator(
            self.schedule(tuple(blocks)), apply_final_frame=apply_final_frame
        )


def assemble_explicit_atom(
    model, calibration, coordinates, *, logical_to_site=None, occupancy=None
):
    """Construct the target-based private atom test path."""
    count = len(coordinates)
    mapping = tuple(range(count)) if logical_to_site is None else tuple(logical_to_site)
    if set(mapping) != set(range(count)) or len(mapping) != count:
        raise ValueError("logical_to_site must be a complete site permutation")
    if occupancy is not None and tuple(occupancy) != (True,) * count:
        raise ValueError("the three-level target is fully occupied")
    target = _Atom3LevelTarget(model, _explicit_arrangement(coordinates))
    program = Program(count)
    refs = tuple(program.quantum_registers[0][index] for index in range(count))
    target_binding = target.bind_program(
        program,
        ResourceLayout(dict(zip(refs, mapping))),
    )
    allocation = _EngineAllocation(target.device_labels, (3,) * count)
    context = _PulseLoweringContext(
        target_binding,
        allocation,
        _ClassicalAllocation.from_program(program),
    )
    return ExplicitAtom3LevelAssembly(
        target=target,
        engine_allocation=allocation,
        implementation_map=default_atom_3level_gate_implementation_map(
            model=model, calibration=calibration
        ),
        adapter=_Atom3LevelQutipAdapter(
            target,
            engine_allocation=allocation,
        ),
        program=program,
        context=context,
    )
