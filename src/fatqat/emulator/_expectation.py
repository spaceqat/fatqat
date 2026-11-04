"""Exact expectation values of observables on a pulse-emulator final state.

The pulse emulator evolves a QuTiP-ordered state: subsystem 0 is the most
significant tensor factor, and ``dims`` lists the subsystem dimensions in that
order. Each observable term is built as a sparse tensor product of local
operators and evaluated with :func:`qutip.expect`, so the evaluation uses the
same tensor ordering as the evolution and shares nothing with the simulator's
kernels.

QuTiP is imported when a value is computed, not when this module loads, so that
importing :mod:`fatqat` stays lightweight.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..observable import NumberOperator, Projector, _Factor, _Factors, _Pauli


def _local_operator(factor: _Factor, dim: int) -> Any:
    """Return the matrix of one observable factor on a ``dim``-level subsystem."""
    import qutip  # pylint: disable=import-outside-toplevel

    if isinstance(factor, _Pauli):
        return {"X": qutip.sigmax, "Y": qutip.sigmay, "Z": qutip.sigmaz}[
            factor.letter
        ]()
    if isinstance(factor, Projector):
        return qutip.projection(dim, factor.level, factor.level)
    if isinstance(factor, NumberOperator):
        return qutip.num(dim)
    raise ValueError(f"unsupported observable factor {factor!r}")


def expectation(
    state: Any, terms: tuple[tuple[float, _Factors], ...], dims: Sequence[int]
) -> float:
    """Return ``<psi|O|psi>`` or ``Tr(rho O)`` for a ket or density operator.

    ``state`` is the emulator's final ``Qobj`` and ``dims`` are the subsystem
    dimensions in QuTiP order. The state's data is reused; only its tensor
    structure is stated explicitly.
    """
    import qutip  # pylint: disable=import-outside-toplevel

    shape = [list(dims), [1] * len(dims) if state.isket else list(dims)]
    state = qutip.Qobj(state.data, dims=shape)
    total = 0.0
    for coefficient, factors in terms:
        local = [qutip.qeye(dim) for dim in dims]
        for axis, factor in factors:
            local[axis] = _local_operator(factor, dims[axis])
        total += coefficient * complex(qutip.expect(qutip.tensor(*local), state)).real
    return total
