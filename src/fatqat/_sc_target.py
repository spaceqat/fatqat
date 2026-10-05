"""Construction constraints shared by the SC compiler and the SC simulator.

`fatqat.compiler` already depends on `fatqat.simulator` because lowering reads
a backend's device sites. If the compiler defined the target and the simulator
imported it, the import cycle would close and both import orders would fail.
The value object and its validation therefore live here, at the top level,
where both packages can import them. `fatqat.compiler` re-exports the class.
The public path is ``fatqat.compiler.SCTarget``. Keep this module dependent on
the standard library only.
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
    """Describe the sites and `CZ` couplings that a compiler can use.

    A target holds two values only: the number of integer-labeled sites, and
    the undirected site pairs that support `CZ`. It has no gate set, no
    implementation map, and no noise model, and it cannot run a program. A
    :py:class:`~fatqat.simulator.SCQubitSimulator` owns those parts.
    Compiling against a target therefore does not determine which backend
    runs the result.

    Both arguments are keyword-only and required. Neither has a default, so
    the target never assumes a grid topology. The constructor reads the
    couplings once. You can therefore pass any iterable, including a
    one-shot generator, and later changes to the container do not affect the
    target. The target stores each edge as ``(low, high)`` and merges
    reversed and repeated edges into their first canonical occurrence. An
    empty ``couplings`` means that the device has no `CZ` edge. It is a
    valid target. Lowering decides later whether the compiler can route a
    given program on it.

    Targets are immutable and reusable. One instance can serve any number of
    compilations, including concurrent ones, and it holds no reference to
    the caller's input. Equality, hashing, and serialization are not part of
    this contract.

    Examples:
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
                strict Python ``int`` greater than zero. The constructor
                rejects ``bool``, ``float``, and NumPy integers and does not
                convert them.
            couplings: Finite iterable of undirected pairs of connected
                device sites. Each element must be a two-element ``tuple``
                of strict ``int`` endpoints in ``range(num_qubits)``.

        Raises:
            TypeError: If the site count or coupling endpoints are not
                integers, or a coupling is not a two-element tuple.
            ValueError: If the site count is not positive, a coupling
                endpoint is out of range, or a coupling connects a site to
                itself.
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
