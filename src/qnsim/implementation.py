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
applied (e.g. `RX(0.3)`) plus the `targets: tuple[RegisterRef, ...]` operand
tuple by keyword, and returns the local matrix — never the surrounding
`AppliedOperation`, and never a feedforward `condition`: condition resolution
happens separately, in the backend. `targets` lets a rule read
`targets[0].register.dim` to build a dimension-dependent matrix (e.g. a
qudit `Shift`/`Clock`/`Sum` gate); a rule whose matrix never depends on
target dimension (every fixed qubit gate here) simply ignores the argument.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import operations as ops
from .operations import Operation
from .registers import RegisterRef

class MatrixImplementation:
    """Base class for a matrix-family implementation rule.

    A rule receives the bare `Operation` instance that was applied (e.g. an
    `RX(0.3)` value) plus the `targets` `RegisterRef` tuple by keyword, and
    returns its local matrix. Most callers never need to subclass this
    directly: `MatrixImplementationMap.register` auto-wraps a plain
    `np.ndarray` (as `FixedMatrix`), a `_DimMatrix`, or a bare
    callable. Subclass and override `__call__` for a stateful or configured
    implementation.
    """

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...]) -> np.ndarray:
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

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...] = ()) -> np.ndarray:
        return self._matrix


class _DimMatrix(MatrixImplementation):
    """A rule whose matrix depends only on the target subsystem dimensions.

    Use this when a gate's matrix is fixed *given* the dimensions of its
    targets but is not itself a single constant matrix (e.g. a qudit `Shift`
    gate, whose permutation matrix depends on `targets[0].register.dim`).
    Unlike `FixedMatrix`, this always reads `targets`, so it cannot be used
    with the `targets=()` default — the caller (backend resolution) always
    supplies the real target tuple.
    """

    def __init__(self, fn: "Callable[[tuple[int, ...]], np.ndarray]") -> None:
        """Store `fn`, called on demand with the target dimensions tuple.

        Args:
            fn: Callable taking a `tuple[int, ...]` of target subsystem
                dimensions (in target order) and returning the local matrix.
        """
        self._fn = fn

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...]) -> np.ndarray:
        dims = tuple(t.register.dim for t in targets)
        return self._fn(dims)


@dataclass(frozen=True)
class ApplyMatrixStep:
    """Resolved local matrix payload consumed by the statevector engine.

    Doubles as the "apply a matrix" entry in a backend execution plan and as the
    payload the engine applies. The matrix is marked read-only after construction
    so this frozen value object cannot be mutated through the NumPy array buffer.

    Attributes:
        matrix: Local operation matrix.
        target_indices: Flat subsystem indices the matrix acts on.
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
        # A rule may hand back a shared/cached array (a `FixedMatrix` copies,
        # but a bare callable need not), so copy before freezing when the array
        # is still writeable - freezing in place would mutate the rule's own
        # object as a side effect. An already read-only array is left as-is.
        if self.matrix.flags.writeable:
            object.__setattr__(self, "matrix", np.array(self.matrix, copy=True))
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


def clock_matrix(dim: int, power: int) -> np.ndarray:
    """Generalized-Pauli clock: diag(omega^(k*power)), omega = e^{2πi/dim}."""
    power %= dim
    omega = np.exp(2j * np.pi / dim)
    return np.diag([omega ** ((k * power) % dim) for k in range(dim)]).astype(complex)


def sum_matrix(dims: tuple[int, ...]) -> np.ndarray:
    """Controlled mod-d add on two equal-dimension subsystems.

    Local index is ``i*d + j`` with operand 0 (control ``i``) the MSB. Maps
    ``|i, j> -> |i, (i + j) mod d>``.
    """
    if len(dims) != 2 or dims[0] != dims[1]:
        raise ValueError(
            f"default Sum requires two equal-dimension targets, got {dims}"
        )
    d = dims[0]
    m = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            m[i * d + (i + j) % d, i * d + j] = 1.0
    return m


def _shift_rule(op: "ops.Shift", targets) -> np.ndarray:
    return shift_matrix(targets[0].register.dim, op.power)


def _clock_rule(op: "ops.Clock", targets) -> np.ndarray:
    return clock_matrix(targets[0].register.dim, op.power)


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

    This is a deliberate scope policy, not a technical limit: rules do receive
    `targets` and could in principle size a matrix from `len(targets)`. But a
    variable-arity operation has no single canonical matrix shape to validate
    a rule's output against, so it stays out of scope for this registry unless
    a concrete variadic-matrix gate need appears.
    """
    if op_cls._num_subsystems is None:
        raise TypeError(
            f"{op_cls.__name__} has variable arity (_num_subsystems is None); "
            "the matrix implementation map only supports fixed-arity operations"
        )


def _callable_wants_targets(rule: Callable) -> bool:
    """True if a bare callable declares a `targets` parameter (or **kwargs).

    A rule is targets-aware if it names a `targets` parameter explicitly, or
    accepts arbitrary keyword arguments (`**kwargs`) and so can absorb a
    `targets=` keyword regardless. If the signature cannot be introspected
    (`inspect.signature` can raise `ValueError` or `TypeError` for some
    C-implemented or otherwise uninspectable callables), the callable is
    conservatively treated as not wanting `targets` and called as `rule(op)`.
    """
    try:
        params = inspect.signature(rule).parameters
    except (ValueError, TypeError):
        return False
    if "targets" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class _CallableMatrixImplementation(MatrixImplementation):
    """Adapts a bare `f(op)` or `f(op, targets)` callable to `MatrixImplementation`."""

    def __init__(self, func: Callable, wants_targets: bool) -> None:
        self._func = func
        self._wants_targets = wants_targets

    def __call__(self, op: Operation, *, targets: tuple[RegisterRef, ...]) -> np.ndarray:
        if self._wants_targets:
            return self._func(op, targets=targets)
        return self._func(op)


def _wrap_rule(
    op_cls: type[Operation],
    rule: "MatrixImplementation | Callable | np.ndarray",
) -> MatrixImplementation:
    """Normalize a `register()` rule argument into a `MatrixImplementation`.

    Accepts an already-built `MatrixImplementation` (returned as-is, e.g. a
    `FixedMatrix` or `_DimMatrix`), a plain `np.ndarray` (wrapped in
    `FixedMatrix`, which only requires it be square with side length >= 2 —
    see `_validate_square_matrix`), or a bare `f(op)`/`f(op, targets)`
    callable (wrapped). Every stored rule is a `MatrixImplementation`
    instance, so `get()` always returns a uniform type regardless of how the
    rule was registered.

    A callable is not arity-checked at registration: a rule that cannot be
    called in its detected `f(op)`/`f(op, targets)` shape raises the first
    time it is used, where the backend wraps it in a `MatrixImplementationError`
    naming the operation. Registration only distinguishes the two shapes (via
    `_callable_wants_targets`) so the call site passes `targets=` iff wanted.

    Raises:
        TypeError: If `rule` is none of the above (e.g. a string or a plain
            object) — checked explicitly here so the error names the
            operation and the bad value.
    """
    if isinstance(rule, MatrixImplementation):
        return rule
    if isinstance(rule, np.ndarray):
        return FixedMatrix(rule)  # square-only validation lives in FixedMatrix
    if not callable(rule):
        raise TypeError(
            f"rule for {op_cls.__name__} must be a MatrixImplementation, "
            f"np.ndarray, or callable, got {rule!r}"
        )
    return _CallableMatrixImplementation(rule, _callable_wants_targets(rule))


class MatrixImplementationMap:
    """Class-keyed registry from operation classes to matrix implementations."""

    def __init__(self) -> None:
        """Create an empty implementation map."""
        self._rules: dict[type[Operation], MatrixImplementation] = {}

    def register(
        self,
        op: Operation | type[Operation],
        rule: "MatrixImplementation | Callable | np.ndarray",
    ) -> None:
        """Register a matrix implementation for an operation.

        Args:
            op: An `Operation` instance (e.g. `qs.ops.X`) or subclass (e.g. a
                custom gate class). Normalized to the operation's class for
                the registry key.
            rule: A `MatrixImplementation` instance (e.g. `FixedMatrix` or
                `_DimMatrix`), a bare `np.ndarray` (wrapped in
                `FixedMatrix`), or a bare callable — either `f(op)` or
                `f(op, targets)`, detected by a parameter literally named
                `targets` (or `**kwargs`) — returning the operation's matrix
                (wrapped automatically).

        Raises:
            TypeError: If `op` is neither an `Operation` instance nor
                subclass, or if its operation class has variable arity. A bare
                callable of the wrong shape is not rejected here; it fails on
                first use (see `_wrap_rule`).
            ValueError: If a bare `np.ndarray` is not square with side
                length >= 2.
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
    m.register(ops.Shift, _shift_rule)
    m.register(ops.Clock, _clock_rule)
    m.register(ops.SumGate, _DimMatrix(sum_matrix))
    return m
