"""Default matrix implementation map: wires built-in gates to their matrices."""

from __future__ import annotations

from .. import operations as ops
from .base import FixedMatrix, ImplementationMap, _DimMatrix
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


def default_matrix_implementation_map() -> ImplementationMap:
    """Build the default matrix implementation map.

    Registers against the public singleton instances (e.g. `ops.X`), not the
    underlying `*Gate` classes: `add()` resolves either to the same
    class key, and the fixed-gate classes are not part of the `fq.ops` public
    surface (see `operations.fixed_gates`).
    """
    m = ImplementationMap()
    m.add(ops.X, FixedMatrix(_X))
    m.add(ops.Y, FixedMatrix(_Y))
    m.add(ops.Z, FixedMatrix(_Z))
    m.add(ops.H, FixedMatrix(_H))
    m.add(ops.I, FixedMatrix(_I))
    m.add(ops.S, FixedMatrix(_S))
    m.add(ops.Sdg, FixedMatrix(_SDG))
    m.add(ops.SX, FixedMatrix(_SX))
    m.add(ops.T, FixedMatrix(_T))
    m.add(ops.Tdg, FixedMatrix(_TDG))
    m.add(ops.CX, FixedMatrix(_CX))
    m.add(ops.CZ, FixedMatrix(_CZ))
    m.add(ops.Swap, FixedMatrix(_SWAP))
    m.add(ops.CY, FixedMatrix(_CY))
    m.add(ops.CS, FixedMatrix(_CS))
    m.add(ops.iSwap, FixedMatrix(_ISWAP))
    m.add(ops.CCX, FixedMatrix(_CCX))
    m.add(ops.CSwap, FixedMatrix(_CSWAP))
    m.add(ops.RX, _rx)
    m.add(ops.RY, _ry)
    m.add(ops.RZ, _rz)
    m.add(ops.Phase, _phase)
    m.add(ops.CPhase, _cphase)
    m.add(ops.Shift, _shift_rule)
    m.add(ops.Clock, _clock_rule)
    m.add(ops.Sum, _DimMatrix(sum_matrix))
    m.add(ops.SwapLevels, _swap_levels_rule)
    m.add(ops.Fourier, _DimMatrix(_fourier_rule))
    m.add(ops.Fourierdg, _DimMatrix(_fourierdg_rule))
    m.add(ops.SubspaceRX, _subspace_rx_rule)
    m.add(ops.SubspaceRY, _subspace_ry_rule)
    m.add(ops.SubspaceRZ, _subspace_rz_rule)
    m.add(ops.CClock, _cclock_rule)
    return m
