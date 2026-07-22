"""Small value object used by the backend's per-run qubit resource map."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from ..flat_layout import FlatResourceLayout
from ..program import Program
from ..registers import RegisterRef

DeviceLabelPolicy = Callable[[RegisterRef, FlatResourceLayout], Hashable]


@dataclass(frozen=True)
class BoundResource:
    """One scalar register reference mapped to engine and device identities.

    ``engine_index`` addresses the simulator's compact flat state.  The
    ``device_label`` is the backend-specific key used for implementation-map
    lookup.  They are equal for identity mappings, but may differ when a
    smaller frontend grid is placed on a larger hardware grid.
    """

    ref: RegisterRef
    engine_index: int
    device_label: Hashable


def _build_qubit_resource_map(
    program: Program,
    flat_layout: FlatResourceLayout,
    device_label_for: DeviceLabelPolicy,
) -> dict[RegisterRef, BoundResource]:
    """Build the complete scalar qubit-resource map for one program run."""
    resources: dict[RegisterRef, BoundResource] = {}
    for register in program.qreg:
        for index in range(register.size):
            ref = register[index]
            resources[ref] = BoundResource(
                ref=ref,
                engine_index=flat_layout.subsystem_index(ref),
                device_label=device_label_for(ref, flat_layout),
            )
    return resources
