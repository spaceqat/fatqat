"""Small, engine-independent target snapshots used by SC routing."""

from dataclasses import dataclass

from .. import operations as ops
from .algorithms.sabre import SiteId


@dataclass(frozen=True, slots=True)
class _SCTarget:
    name: str
    sites: tuple[SiteId, ...]
    couplings: frozenset[tuple[SiteId, SiteId]]
    profile: str


def _simulator_target(backend, profile: str) -> _SCTarget:
    return _SCTarget(
        type(backend).__name__,
        backend.device_sites,
        backend.implementation_map.device_operands_for(ops.CZ),
        profile,
    )
