"""Value objects shared by the internal ZAP algorithm implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .architecture import architecture_sites
from .router import Router
from .scheduler import Scheduler


@dataclass(frozen=True, slots=True)
class ZapInteraction:
    """One identified one- or two-atom ZAP interaction."""

    operation_id: str
    atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a non-empty string")
        if len(self.atoms) not in (1, 2):
            raise ValueError("interaction must contain one or two atoms")
        if any(type(atom) is not int or atom < 0 for atom in self.atoms):
            raise ValueError("atom indices must be non-negative integers")
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("two-atom interaction endpoints must be distinct")


@dataclass(frozen=True, slots=True)
class ZapTrace:
    """Ordered ZAP instruction trace associated with an explicit atom count."""

    atom_count: int
    instructions: tuple[Mapping[str, object], ...]


def _normalize_coordinate(coordinate: object) -> int:
    if type(coordinate) is int:
        return coordinate
    if (
        type(coordinate) is float
        and math.isfinite(coordinate)
        and coordinate.is_integer()
    ):
        return int(coordinate)
    raise ValueError("initial_mapping coordinates must be finite integers")


def compile_interactions(
    interactions: tuple[ZapInteraction, ...],
    architecture: Mapping[str, object],
    *,
    atom_count: int,
    initial_mapping: tuple[tuple[float, float], ...] = (),
    scheduling_strategy: str = "asap_joint",
) -> ZapTrace:
    """Compile identified dense-atom interactions into a ZAP instruction trace."""
    if type(atom_count) is not int or atom_count <= 0:
        raise ValueError("atom_count must be a positive integer")
    if any(atom >= atom_count for item in interactions for atom in item.atoms):
        raise ValueError("interaction names an atom outside atom_count")
    if scheduling_strategy != "asap_joint":
        raise ValueError("the library API currently supports only asap_joint")
    if len({item.operation_id for item in interactions}) != len(interactions):
        raise ValueError("operation_id values must be unique")

    normalized_initial_mapping = [
        (_normalize_coordinate(x), _normalize_coordinate(y)) for x, y in initial_mapping
    ]

    storage_sites = architecture_sites(architecture, "storage_zones")
    if atom_count > len(storage_sites):
        raise ValueError(
            f"atom_count requires {atom_count} atoms, but architecture defines only "
            f"{len(storage_sites)} storage traps"
        )
    entanglement_sites = architecture_sites(architecture, "entanglement_zones")

    g_q = [
        (item.atoms[0], item.atoms[0]) if len(item.atoms) == 1 else item.atoms
        for item in interactions
    ]
    results_code = {
        "n_q": atom_count,
        "n_1q_gate": sum(len(item.atoms) == 1 for item in interactions),
        "n_2q_gate": sum(len(item.atoms) == 2 for item in interactions),
        "stages": {},
        "instructions": [],
    }
    scheduler = Scheduler(
        g_q=g_q,
        operation_ids=tuple(item.operation_id for item in interactions),
        results_code=results_code,
    )
    scheduler.asap_joint()

    stage_data = results_code["stages"]["stage"]
    list_full_gates = [
        [g_q[index] for index in stage] for stage in scheduler.list_scheduling
    ]
    list_operation_ids = [
        stage_data[index]["operation_ids"]
        for index in range(results_code["stages"]["num_stage"])
    ]
    router = Router(
        slm_sites=[storage_sites, entanglement_sites],
        results_code=results_code,
        list_full_gates=list_full_gates,
        list_operation_ids=list_operation_ids,
        qubit_mapping=normalized_initial_mapping,
        architecture=architecture,
    )
    return ZapTrace(atom_count, tuple(router.results_code["instructions"]))
