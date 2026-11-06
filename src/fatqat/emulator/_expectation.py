"""Exact expectation values of observables on a pulse-emulator final state.

The pulse emulator evolves a QuTiP-ordered state: subsystem 0 is the most
significant tensor factor, and ``dims`` lists the subsystem dimensions in that
order. Each observable is built once as a sparse operator and evaluated with
:func:`qutip.expect`, so the evaluation uses the same tensor ordering as the
evolution and shares nothing with the simulator's kernels.

Building the operator is the expensive part: it costs more than evaluating it.
So the two are separate steps, and the caller keeps an observable's operator
for as long as it evaluates that observable, across every entry of a parameter
sweep.

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


def _term_operator(coefficient: float, factors: _Factors, dims: Sequence[int]) -> Any:
    """Return one term's tensor product, scaled by its coefficient.

    Each run of subsystems the term does not name becomes one identity block,
    so the product takes one Kronecker step per named factor rather than one
    per subsystem. The coefficient scales the first block, which is small,
    instead of the full product.
    """
    import qutip  # pylint: disable=import-outside-toplevel

    named = dict(factors)
    blocks = []
    identity: list[int] = []
    for axis, dim in enumerate(dims):
        if axis not in named:
            identity.append(dim)
            continue
        if identity:
            blocks.append(qutip.qeye(identity))
            identity = []
        blocks.append(_local_operator(named[axis], dim))
    if identity:
        blocks.append(qutip.qeye(identity))
    blocks[0] = coefficient * blocks[0]
    return qutip.tensor(*blocks)


def observable_operator(
    terms: tuple[tuple[float, _Factors], ...], dims: Sequence[int]
) -> Any:
    """Return an observable's operator on subsystems of dimensions ``dims``.

    ``terms`` must not be empty. Every factor is Hermitian and every
    coefficient real, so the operator is marked Hermitian, which lets
    :func:`qutip.expect` return a real value without checking.
    """
    import qutip  # pylint: disable=import-outside-toplevel

    first, *rest = terms
    operator = _term_operator(*first, dims)
    for coefficient, factors in rest:
        operator = operator + _term_operator(coefficient, factors, dims)
    return qutip.Qobj(operator.data, dims=operator.dims, copy=False, isherm=True)


def with_dims(state: Any, dims: Sequence[int]) -> Any:
    """Return ``state`` with the subsystem structure ``dims`` stated explicitly.

    The emulator's state already has it in the usual case, and is then
    returned as is. Otherwise the data is shared, not copied, and a density
    operator's Hermitian status is carried over, so it is settled once here
    rather than by every evaluation.
    """
    import qutip  # pylint: disable=import-outside-toplevel

    shape = [list(dims), [1] * len(dims) if state.isket else list(dims)]
    if state.dims == shape:
        return state
    return qutip.Qobj(
        state.data,
        dims=shape,
        copy=False,
        isherm=None if state.isket else state.isherm,
    )


def expectation(state: Any, operator: Any) -> float:
    """Return ``<psi|O|psi>`` or ``Tr(rho O)`` for a ket or density operator.

    ``state`` must carry the operator's subsystem structure; see
    :func:`with_dims`.
    """
    import qutip  # pylint: disable=import-outside-toplevel

    return float(complex(qutip.expect(operator, state)).real)
