"""Class-keyed matrix implementations and the flat payload the engine consumes.

A matrix implementation maps an operation to its local matrix (physics only).
The backend pairs that matrix with layout-resolved target indices to build an
``ApplyMatrixStep`` — the plain data container the statevector engine reads
directly.

Local matrix convention (binding for every entry in this module):
    - ``AppliedOperation.targets`` operand order defines the local
      tensor-factor order; ``targets[0]`` is the local most-significant bit
      (MSB), ``targets[-1]`` the local least-significant bit (LSB). See
      ``engine._apply_matrix`` for the little-endian contraction this feeds.
    - For every controlled gate below (``CX``, ``CZ``, ``CY``, and the
      controlled gates added in later batches), the control operand(s) come
      first and the target operand(s) come last — operand 0 (and operand 1
      for doubly-controlled gates) is the control, occupying the local MSB
      position(s).

A matrix implementation rule receives the bare `Operation` instance that was
applied (e.g. `RX(0.3)`), never the surrounding `AppliedOperation` — target
and feedforward-condition resolution both happen separately, in the backend.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import operations as ops
from .operations import Operation

class MatrixImplementation:
    """Base class for a matrix-family implementation rule.

    A rule receives the bare `Operation` instance that was applied (e.g. an
    `RX(0.3)` value) and returns its local matrix. Most callers never need to
    subclass this directly: `MatrixImplementationMap.register` auto-wraps a
    plain `np.ndarray` (as `FixedMatrix`) or a bare callable. Subclass and
    override `__call__` for a stateful or configured implementation.
    """

    def __call__(self, op: Operation) -> np.ndarray:
        raise NotImplementedError


def _validate_square_matrix(matrix: np.ndarray) -> None:
    """Raise `ValueError` unless `matrix` is square with side length >= 2.

    Deliberately does not require a power-of-two side length: `FixedMatrix`
    has no way to know what dimension its caller intends (it never sees the
    target operation or register), and a fixed-dimension restriction here
    would reject legitimate non-qubit matrices (e.g. a qutrit's dim=3 gate)
    with no compensating safety benefit — the arity-aware shape check against
    a specific operation happens separately, in `_wrap_rule`.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"matrix side length must be >= 2, got {n}")


class FixedMatrix(MatrixImplementation):
    """A constant matrix, independent of the applied operation's fields."""

    def __init__(self, matrix: np.ndarray) -> None:
        """Copy, validate, and freeze `matrix` as this rule's constant value.

        Args:
            matrix: Square matrix with side length >= 2. Copied on
                construction, so later mutation of the caller's array does
                not affect this rule.

        Raises:
            ValueError: If `matrix` is not square or its side length is < 2.
        """
        matrix = np.array(matrix, dtype=complex, copy=True)
        _validate_square_matrix(matrix)
        matrix.flags.writeable = False
        self._matrix = matrix

    def __call__(self, op: Operation) -> np.ndarray:
        return self._matrix


@dataclass(frozen=True)
class ApplyMatrixStep:
    """Resolved local matrix payload consumed by the statevector engine.

    Doubles as the "apply a matrix" entry in a backend execution plan and as the
    payload the engine applies. The matrix is marked read-only after construction
    so this frozen value object cannot be mutated through the NumPy array buffer.

    Attributes:
        matrix: Local operation matrix.
        target_indices: Flat qubit indices the matrix acts on.
        condition: Optional feedforward guard as lowered ``(clbit_index, value)``
            AND-terms. ``None`` means unconditional. The engine ignores this
            field; the backend's per-shot loop evaluates it.
    """

    matrix: np.ndarray
    target_indices: tuple[int, ...]
    condition: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        # The engine consumes the matrix read-only; lock it so this frozen
        # dataclass is truly immutable (Python cannot freeze array contents).
        self.matrix.flags.writeable = False


def shift_matrix(dim: int, power: int) -> np.ndarray:
    """Generalized-Pauli shift: ``|k> -> |(k + power) mod dim>``.

    Returns a ``dim x dim`` permutation matrix. ``power`` is reduced modulo
    ``dim``, so ``shift_matrix(3, 5)`` equals ``shift_matrix(3, 2)``.
    """
    power %= dim
    m = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        m[(k + power) % dim, k] = 1.0
    return m


# Module-level constant matrices (reused; do not rebuild per call).
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_I = np.eye(2, dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
# 2-qubit fixed gates (see module docstring for the control/target convention).
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
_CY = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex
)
_CS = np.diag([1, 1, 1, 1j]).astype(complex)
_ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)
# 3-qubit fixed gates. Basis index bits are (operand0, operand1, operand2)
# from MSB to LSB, matching the control-first convention above.
_CCX = np.eye(8, dtype=complex)
_CCX[[6, 7]] = _CCX[[7, 6]]  # swap |110> <-> |111>: flip target iff both controls=1

_CSWAP = np.eye(8, dtype=complex)
_CSWAP[[5, 6]] = _CSWAP[[6, 5]]  # swap |101> <-> |110>: exchange targets iff control=1


def _rx(op: ops.RX) -> np.ndarray:
    """Build the RX matrix from the operation's angle."""
    theta = op.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(op: ops.RY) -> np.ndarray:
    """Build the RY matrix from the operation's angle."""
    theta = op.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(op: ops.RZ) -> np.ndarray:
    """Build the RZ matrix from the operation's angle."""
    theta = op.theta
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def _phase(op: ops.Phase) -> np.ndarray:
    """Build the Phase matrix from the operation's angle."""
    theta = op.theta
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def _cphase(op: ops.CPhase) -> np.ndarray:
    """Build the CPhase matrix from the operation's angle."""
    theta = op.theta
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def _resolve_operation_class(op: Operation | type[Operation]) -> type[Operation]:
    """Normalize an `Operation` instance or subclass to its registry key.

    Accepts either an `Operation` instance (e.g. `qs.ops.X`) or an `Operation`
    subclass (e.g. a custom gate class) and returns the class to key the
    registry by. Applying `type(...)` unconditionally would be wrong for the
    class case: `type(MyGate)` is the metaclass `type`, not `MyGate`.
    """
    if isinstance(op, Operation):
        return type(op)
    if isinstance(op, type) and issubclass(op, Operation):
        return op
    raise TypeError(f"expected an Operation instance or subclass, got {op!r}")


def _require_fixed_arity(op_cls: type[Operation]) -> None:
    """Raise `TypeError` if `op_cls` has variable arity (`_num_subsystems is None`).

    A matrix map rule receives only the bare `Operation` instance, never the
    application's target count, so there is no way for a rule to know what
    size matrix to build for a variable-arity operation. Such operations are
    out of scope for this registry.
    """
    if op_cls._num_subsystems is None:
        raise TypeError(
            f"{op_cls.__name__} has variable arity (_num_subsystems is None); "
            "the matrix implementation map only supports fixed-arity operations"
        )


_ARITY_CHECK_SENTINEL = object()


class _CallableMatrixImplementation(MatrixImplementation):
    """Adapts a bare `Operation -> np.ndarray` callable to `MatrixImplementation`."""

    def __init__(self, func: Callable[[Operation], np.ndarray]) -> None:
        self._func = func

    def __call__(self, op: Operation) -> np.ndarray:
        return self._func(op)


def _check_callable_arity(op_cls: type[Operation], rule: Callable) -> None:
    """Raise `TypeError` if `rule` cannot be called with one positional argument.

    Uses `inspect.signature(rule).bind(...)` rather than counting parameters:
    this accepts a `*args`-only callable or one with an optional second
    parameter (one positional slot is enough) and rejects a required
    keyword-only parameter (no positional slot can satisfy it), matching
    what `rule(op)` will actually do at call time. If the signature cannot
    be introspected at all (`inspect.signature` can raise either `ValueError`
    or `TypeError` for some C-implemented or otherwise uninspectable
    callables — a different `TypeError` than the one this function itself
    raises below for a genuine arity mismatch), the check is skipped and the
    callable is accepted — this is a best-effort early warning, not a hard
    gate.
    """
    try:
        signature = inspect.signature(rule)
    except (ValueError, TypeError):
        return
    try:
        signature.bind(_ARITY_CHECK_SENTINEL)
    except TypeError:
        raise TypeError(
            f"rule for {op_cls.__name__} must accept one positional argument "
            f"(the Operation instance), got signature {signature}"
        ) from None


def _wrap_rule(
    op_cls: type[Operation],
    rule: "MatrixImplementation | Callable[[Operation], np.ndarray] | np.ndarray",
) -> MatrixImplementation:
    """Normalize a `register()` rule argument into a `MatrixImplementation`.

    Accepts an already-built `MatrixImplementation` (returned as-is), a plain
    `np.ndarray` (wrapped in `FixedMatrix`, shape-validated against `op_cls`'s
    fixed arity), or a bare callable (arity-checked and wrapped). Every
    stored rule is a `MatrixImplementation` instance, so `get()` always
    returns a uniform type regardless of how the rule was registered.

    Raises:
        TypeError: If `rule` is none of the above (e.g. a string or a plain
            object) — checked explicitly here so the error names the
            operation and the bad value, rather than surfacing whatever raw
            `TypeError` `inspect.signature` happens to raise for a
            non-callable.
    """
    if isinstance(rule, MatrixImplementation):
        return rule
    if isinstance(rule, np.ndarray):
        n = op_cls._num_subsystems
        expected_shape = (2**n, 2**n)
        if rule.shape != expected_shape:
            raise ValueError(
                f"matrix for {op_cls.__name__} must have shape {expected_shape}, "
                f"got {rule.shape}"
            )
        return FixedMatrix(rule)
    if not callable(rule):
        raise TypeError(
            f"rule for {op_cls.__name__} must be a MatrixImplementation, "
            f"np.ndarray, or callable, got {rule!r}"
        )
    _check_callable_arity(op_cls, rule)
    return _CallableMatrixImplementation(rule)


class MatrixImplementationMap:
    """Class-keyed registry from operation classes to matrix implementations."""

    def __init__(self) -> None:
        """Create an empty implementation map."""
        self._rules: dict[type[Operation], MatrixImplementation] = {}

    def register(
        self,
        op: Operation | type[Operation],
        rule: "MatrixImplementation | Callable[[Operation], np.ndarray] | np.ndarray",
    ) -> None:
        """Register a matrix implementation for an operation.

        Args:
            op: An `Operation` instance (e.g. `qs.ops.X`) or subclass (e.g. a
                custom gate class). Normalized to the operation's class for
                the registry key.
            rule: A `MatrixImplementation` instance, a bare `np.ndarray`
                (wrapped in `FixedMatrix`), or a bare callable taking the
                operation and returning its matrix (wrapped automatically).

        Raises:
            TypeError: If `op` is neither an `Operation` instance nor
                subclass; if its operation class has variable arity; or if a
                bare callable cannot accept one positional argument.
            ValueError: If a bare `np.ndarray` does not match the shape
                required by `op`'s arity.
        """
        op_cls = _resolve_operation_class(op)
        _require_fixed_arity(op_cls)
        self._rules[op_cls] = _wrap_rule(op_cls, rule)

    def unregister(self, op: Operation | type[Operation]) -> None:
        """Remove a registered matrix implementation, if present.

        Args:
            op: An `Operation` instance or subclass to remove. Removing an
                operation that was never registered is a no-op.
        """
        op_cls = _resolve_operation_class(op)
        self._rules.pop(op_cls, None)

    def get(self, op: Operation | type[Operation]) -> MatrixImplementation | None:
        """Return the matrix implementation registered for an operation, if any.

        Always a `MatrixImplementation` instance regardless of what was
        registered — a bare callable is wrapped, a bare ndarray becomes a
        `FixedMatrix`.

        Args:
            op: An `Operation` instance or subclass.
        """
        op_cls = _resolve_operation_class(op)
        return self._rules.get(op_cls)

    def copy(self) -> "MatrixImplementationMap":
        """Return a new map with an independent copy of this map's registrations.

        Rule objects themselves are shared (not deep-copied) between the
        original and the copy — rules are expected to be immutable or
        self-contained, so sharing them across independent map copies is
        safe. Mutating one map's registrations (`register`/`unregister`)
        never affects the other.
        """
        clone = MatrixImplementationMap()
        clone._rules = dict(self._rules)
        return clone


def default_implementation_map() -> MatrixImplementationMap:
    """Build the default matrix implementation map."""
    m = MatrixImplementationMap()
    m.register(ops.XGate, FixedMatrix(_X))
    m.register(ops.YGate, FixedMatrix(_Y))
    m.register(ops.ZGate, FixedMatrix(_Z))
    m.register(ops.HGate, FixedMatrix(_H))
    m.register(ops.IGate, FixedMatrix(_I))
    m.register(ops.SGate, FixedMatrix(_S))
    m.register(ops.SdgGate, FixedMatrix(_SDG))
    m.register(ops.TGate, FixedMatrix(_T))
    m.register(ops.TdgGate, FixedMatrix(_TDG))
    m.register(ops.CXGate, FixedMatrix(_CX))
    m.register(ops.CZGate, FixedMatrix(_CZ))
    m.register(ops.SwapGate, FixedMatrix(_SWAP))
    m.register(ops.CYGate, FixedMatrix(_CY))
    m.register(ops.CSGate, FixedMatrix(_CS))
    m.register(ops.iSwapGate, FixedMatrix(_ISWAP))
    m.register(ops.CCXGate, FixedMatrix(_CCX))
    m.register(ops.CSwapGate, FixedMatrix(_CSWAP))
    m.register(ops.RX, _rx)
    m.register(ops.RY, _ry)
    m.register(ops.RZ, _rz)
    m.register(ops.Phase, _phase)
    m.register(ops.CPhase, _cphase)
    return m
