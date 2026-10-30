"""Built-in matrix and channel implementation rules."""

from __future__ import annotations

from .. import operations as ops
from .._backends.steps import BuiltinKernelKey as K
from .matrix_map import (
    FixedMatrix,
    MatrixImplementationMap,
    _DimMatrix,
    _BatchedBuilder,
    _KeyedImplementation,
    _wrap_rule,
)
from .base_map import _resolve_operation_class
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
    _beam_splitter_rule,
    _cclock_rule,
    _clock_rule,
    _cphase,
    _fourier_rule,
    _fourierdg_rule,
    _phase,
    _phase_shifter_rule,
    _u,
    _u1,
    _u2,
    _u3,
    _rx,
    _ry,
    _rz,
    _shift_rule,
    _subspace_rx_rule,
    _subspace_ry_rule,
    _subspace_rz_rule,
    _swap_levels_rule,
    _cphase_batch_rule,
    _phase_batch_rule,
    _rx_batch_rule,
    _ry_batch_rule,
    _rz_batch_rule,
    _subspace_rx_batch_rule,
    _subspace_ry_batch_rule,
    _subspace_rz_batch_rule,
    _u_batch_rule,
    _u1_batch_rule,
    _u2_batch_rule,
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
    (ops.U, _u, K.U),
    (ops.U1, _u1, K.U1),
    (ops.U2, _u2, K.U2),
    (ops.U3, _u3, K.U3),
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
    (ops.PhaseShifter, _phase_shifter_rule, K.PHASE_SHIFTER),
    (ops.BeamSplitter, _beam_splitter_rule, K.BEAM_SPLITTER),
)

# Batched twins of the parametric built-in rules, used by sweep replay. Each
# reads the same fields as its scalar rule in `_DEFAULT_RULES`.
_BATCHED_RULES: dict[type[ops.Operation], _BatchedBuilder] = {
    ops.RX: _rx_batch_rule,
    ops.RY: _ry_batch_rule,
    ops.RZ: _rz_batch_rule,
    ops.Phase: _phase_batch_rule,
    ops.U: _u_batch_rule,
    ops.U1: _u1_batch_rule,
    ops.U2: _u2_batch_rule,
    ops.U3: _u_batch_rule,
    ops.CPhase: _cphase_batch_rule,
    ops.SubspaceRX: _subspace_rx_batch_rule,
    ops.SubspaceRY: _subspace_ry_batch_rule,
    ops.SubspaceRZ: _subspace_rz_batch_rule,
}


def default_matrix_implementation_map() -> MatrixImplementationMap:
    """Return a new map containing built-in matrix gates and finite channel rules.

    Each call returns an independent map. Add, replace, or remove rules on the
    returned value without changing later calls or another backend's map.

    Returns:
        An implementation map covering built-in matrix operations and the
        Depolarizing, AmplitudeDamping, PhaseDamping, PauliChannel, and
        TransitionRelaxation channels.
    """
    m = MatrixImplementationMap()
    _add_default_channel_rules(m)
    for op, rule, key in _DEFAULT_RULES:
        op_cls = _resolve_operation_class(op)
        implementation = _KeyedImplementation(
            _wrap_rule(op_cls, rule),
            key,
            op_cls=op_cls,
            batch=_BATCHED_RULES.get(op_cls),
        )
        m.add(op, implementation)
    return m


def _add_default_channel_rules(implementations: MatrixImplementationMap) -> None:
    """Include the standard finite channels in a simulator profile's map."""
    from ..noise.catalog import (
        AmplitudeDamping,
        Depolarizing,
        PauliChannel,
        PhaseDamping,
        amplitude_damping_rule,
        depolarizing_rule,
        pauli_channel_rule,
        phase_damping_rule,
    )
    from ..noise.transition_relaxation import (
        TransitionRelaxation,
        transition_relaxation_rule,
    )

    implementations.add(Depolarizing, depolarizing_rule)
    implementations.add(AmplitudeDamping, amplitude_damping_rule)
    implementations.add(PhaseDamping, phase_damping_rule)
    implementations.add(PauliChannel, pauli_channel_rule)
    implementations.add(TransitionRelaxation, transition_relaxation_rule)
