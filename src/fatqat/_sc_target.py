"""Construction constraints shared by the SC compiler and the SC simulator.

`fatqat.compiler` already depends on `fatqat.simulator`: lowering reads a
backend's device sites. Defining the target inside the compiler and importing
it back from the simulator would close that cycle and break both import
orders, so the value object and its validation live here, at the top level,
where both packages can reach them. `fatqat.compiler` re-exports the class;
the public path is ``fatqat.compiler.SCTarget``. This module deliberately
depends on the standard library only.
"""

from __future__ import annotations

from collections.abc import Iterable


def _validate_num_qubits(num_qubits: int) -> int:
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer")
    return num_qubits


def _normalize_couplings(
    num_qubits: int, couplings: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge in couplings:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError("couplings must contain two-integer tuples")
        first, second = edge
        if type(first) is not int or type(second) is not int:
            raise TypeError("coupling endpoints must be integers")
        if not 0 <= first < num_qubits or not 0 <= second < num_qubits:
            raise ValueError("coupling endpoint is outside device_sites")
        if first == second:
            raise ValueError("coupling endpoints must be distinct")
        canonical = (min(first, second), max(first, second))
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return tuple(normalized)


class SCTarget:
    """Describe what a superconducting device offers a compiler.

    A target carries exactly two things: how many integer-labeled sites
    exist, and which undirected pairs of them support `CZ`. It has no gate
    set, no implementation map, no noise model and no way to run anything -
    a :py:class:`~fatqat.simulator.SCQubitSimulator` still owns all of that.
    Compiling against a target therefore says nothing about which backend
    will execute the result.

    Both fields are keyword-only and required; neither has a default, so no
    grid topology is ever assumed on the caller's behalf. Couplings are read
    once during construction, which makes any iterable - including a
    one-shot generator - safe to pass and leaves the target unaffected by
    later edits to the container. Each edge is stored as ``(low, high)``,
    with reversed and repeated edges merged into the first canonical
    occurrence. An empty ``couplings`` means the device has no `CZ` edge at
    all, which is a valid target; whether a given program can be routed on
    it is decided later, during lowering.

    Targets are immutable and reusable: one instance can back any number of
    compilations, concurrently, and holds no reference to the caller's
    input. Equality, hashing and serialization are not part of this
    contract.

    Example:
        >>> from fatqat.compiler import SCTarget
        >>> target = SCTarget(num_qubits=3, couplings=[(1, 0), (1, 2), (0, 1)])
        >>> target.num_qubits
        3
        >>> target.couplings
        ((0, 1), (1, 2))
    """

    __slots__ = ("_num_qubits", "_couplings")

    def __init__(
        self, *, num_qubits: int, couplings: Iterable[tuple[int, int]]
    ) -> None:
        """Create a compilation target from a site count and its CZ edges.

        Args:
            num_qubits: Number of integer-labeled device sites. Must be a
                strict Python ``int`` greater than zero; ``bool``, ``float``
                and NumPy integers are rejected rather than coerced.
            couplings: Finite iterable of undirected pairs of connected
                device sites. Each element must be a two-element ``tuple``
                of strict ``int`` endpoints in ``range(num_qubits)``.

        Raises:
            TypeError: If the site count or coupling endpoints are not
                integers, or a coupling is not a two-element tuple.
            ValueError: If the site count is not positive, or a coupling
                endpoint is out of range or repeats its own site.
        """
        self._num_qubits = _validate_num_qubits(num_qubits)
        self._couplings = _normalize_couplings(self._num_qubits, couplings)

    @property
    def num_qubits(self) -> int:
        """Return the number of integer-labeled device sites."""
        return self._num_qubits

    @property
    def couplings(self) -> tuple[tuple[int, int], ...]:
        """Return the canonical undirected `CZ` edges, each as ``(low, high)``."""
        return self._couplings
