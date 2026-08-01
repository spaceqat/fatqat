"""Default matrix implementation map: wires built-in gates to their matrices.

Every default registration is wrapped in `_KeyedImplementation`, attaching the
gate's `BuiltinKernelKey` so engines can dispatch specialized kernels by
declared identity instead of inspecting matrices (see ``backends.steps``).
This is the *only* place keys are attached: custom rules, arrays, and
device-specific overrides registered by users or device backends stay
``None``-keyed by construction.
"""

from __future__ import annotations

from .. import operations as ops
from ..backends.steps import BuiltinKernelKey as K
from .base import (
    FixedMatrix,
    ImplementationMap,
    _DimMatrix,
    _KeyedImplementation,
    _wrap_rule,
)
from ._operation_registry import _resolve_operation_class
from .matrices import (
    _CCX,
    _CS,
    _CSWAP,
    _CX,
    _CY,
    _CZ,
    _H,
    _I,
    _ISWAP,
    _S,
    _SDG,
    _SWAP,
    _SX,
    _T,
    _TDG,
    _X,
    _Y,
    _Z,
    _cclock_rule,
    _clock_rule,
    _cphase,
    _fourier_rule,
    _fourierdg_rule,
    _phase,
    _rx,
    _ry,
    _rz,
    _shift_rule,
    _subspace_rx_rule,
    _subspace_ry_rule,
    _subspace_rz_rule,
    _swap_levels_rule,
    sum_matrix,
)

# (gate singleton, rule, canonical kernel key) - one row per built-in gate.
_DEFAULT_RULES = (
    (ops.X, FixedMatrix(_X), K.X),
    (ops.Y, FixedMatrix(_Y), K.Y),
    (ops.Z, FixedMatrix(_Z), K.Z),
    (ops.H, FixedMatrix(_H), K.H),
    (ops.I, FixedMatrix(_I), K.I),
    (ops.S, FixedMatrix(_S), K.S),
    (ops.Sdg, FixedMatrix(_SDG), K.SDG),
    (ops.SX, FixedMatrix(_SX), K.SX),
    (ops.T, FixedMatrix(_T), K.T),
    (ops.Tdg, FixedMatrix(_TDG), K.TDG),
    (ops.CX, FixedMatrix(_CX), K.CX),
    (ops.CZ, FixedMatrix(_CZ), K.CZ),
    (ops.Swap, FixedMatrix(_SWAP), K.SWAP),
    (ops.CY, FixedMatrix(_CY), K.CY),
    (ops.CS, FixedMatrix(_CS), K.CS),
    (ops.iSwap, FixedMatrix(_ISWAP), K.ISWAP),
    (ops.CCX, FixedMatrix(_CCX), K.CCX),
    (ops.CSwap, FixedMatrix(_CSWAP), K.CSWAP),
    (ops.RX, _rx, K.RX),
    (ops.RY, _ry, K.RY),
    (ops.RZ, _rz, K.RZ),
    (ops.Phase, _phase, K.PHASE),
    (ops.CPhase, _cphase, K.CPHASE),
    (ops.Shift, _shift_rule, K.SHIFT),
    (ops.Clock, _clock_rule, K.CLOCK),
    (ops.Sum, _DimMatrix(sum_matrix), K.SUM),
    (ops.SwapLevels, _swap_levels_rule, K.SWAP_LEVELS),
    (ops.Fourier, _DimMatrix(_fourier_rule), K.FOURIER),
    (ops.InverseFourier, _DimMatrix(_fourierdg_rule), K.FOURIERDG),
    (ops.SubspaceRX, _subspace_rx_rule, K.SUBSPACE_RX),
    (ops.SubspaceRY, _subspace_ry_rule, K.SUBSPACE_RY),
    (ops.SubspaceRZ, _subspace_rz_rule, K.SUBSPACE_RZ),
    (ops.CClock, _cclock_rule, K.CCLOCK),
)


def default_matrix_implementation_map() -> ImplementationMap:
    """Build the default matrix implementation map.

    Registers against the public singleton instances (e.g. `ops.X`), not the
    underlying `*Gate` classes: `add()` resolves either to the same
    class key, and the fixed-gate classes are not part of the
    `fatqat.operations` public
    surface (see `operations.fixed_gates`). Each rule is normalized (bare
    callables wrapped, exactly as `add()` would) and then keyed with its
    gate's canonical `BuiltinKernelKey`.
    """
    m = ImplementationMap()
    for op, rule, key in _DEFAULT_RULES:
        op_cls = _resolve_operation_class(op)
        m.add(op, _KeyedImplementation(_wrap_rule(op_cls, rule), key))
    return m
