"""Default matrix implementation map: wires built-in gates to their matrices."""

from __future__ import annotations

from .. import operations as ops
from .base import FixedMatrix, MatrixImplementationMap, _DimMatrix
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
    _T,
    _TDG,
    _X,
    _Y,
    _Z,
    _clock_rule,
    _cphase,
    _phase,
    _rx,
    _ry,
    _rz,
    _shift_rule,
    sum_matrix,
)


def default_implementation_map() -> MatrixImplementationMap:
    """Build the default matrix implementation map.

    Registers against the public singleton instances (e.g. `ops.X`), not the
    underlying `*Gate` classes: `register()` resolves either to the same
    class key, and the fixed-gate classes are not part of the `qs.ops` public
    surface (see `operations.fixed_gates`).
    """
    m = MatrixImplementationMap()
    m.register(ops.X, FixedMatrix(_X))
    m.register(ops.Y, FixedMatrix(_Y))
    m.register(ops.Z, FixedMatrix(_Z))
    m.register(ops.H, FixedMatrix(_H))
    m.register(ops.I, FixedMatrix(_I))
    m.register(ops.S, FixedMatrix(_S))
    m.register(ops.Sdg, FixedMatrix(_SDG))
    m.register(ops.T, FixedMatrix(_T))
    m.register(ops.Tdg, FixedMatrix(_TDG))
    m.register(ops.CX, FixedMatrix(_CX))
    m.register(ops.CZ, FixedMatrix(_CZ))
    m.register(ops.Swap, FixedMatrix(_SWAP))
    m.register(ops.CY, FixedMatrix(_CY))
    m.register(ops.CS, FixedMatrix(_CS))
    m.register(ops.iSwap, FixedMatrix(_ISWAP))
    m.register(ops.CCX, FixedMatrix(_CCX))
    m.register(ops.CSwap, FixedMatrix(_CSWAP))
    m.register(ops.RX, _rx)
    m.register(ops.RY, _ry)
    m.register(ops.RZ, _rz)
    m.register(ops.Phase, _phase)
    m.register(ops.CPhase, _cphase)
    m.register(ops.Shift, _shift_rule)
    m.register(ops.Clock, _clock_rule)
    m.register(ops.Sum, _DimMatrix(sum_matrix))
    return m
