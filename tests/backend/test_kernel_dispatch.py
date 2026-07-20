"""Key-driven numba kernel dispatch: spec/content agreement and parity."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.backends.steps import ApplyMatrixStep, BuiltinKernelKey
from fatqat.implementation import default_matrix_implementation_map

numba = pytest.importorskip("numba")

# pylint: disable=wrong-import-position  # these need the importorskip above
from fatqat.simulator.nb import (
    _classify_matrix,
    _DENSE,
    _KERNEL_SPECS,
    NumbaSVSimulator,
)
from fatqat.simulator.np import NumpySVSimulator

# pylint: enable=wrong-import-position

# One representative applied instance per default-map gate, with generic
# parameter values (structure at special angles may be *more* special than
# the declared code - RX(pi) is a permutation - which is exactly what the
# spec-vs-content comparison below is careful about).
_QUBIT = fq.QuantumRegister(3)
_QUTRIT = fq.QuantumRegister(2, dim=3)
_GATE_CASES = [
    (fq.ops.X, (_QUBIT[0],)),
    (fq.ops.Y, (_QUBIT[0],)),
    (fq.ops.Z, (_QUBIT[0],)),
    (fq.ops.H, (_QUBIT[0],)),
    (fq.ops.I, (_QUBIT[0],)),
    (fq.ops.S, (_QUBIT[0],)),
    (fq.ops.Sdg, (_QUBIT[0],)),
    (fq.ops.SX, (_QUBIT[0],)),
    (fq.ops.T, (_QUBIT[0],)),
    (fq.ops.Tdg, (_QUBIT[0],)),
    (fq.ops.CX, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CZ, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.Swap, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CY, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CS, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.iSwap, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CCX, (_QUBIT[0], _QUBIT[1], _QUBIT[2])),
    (fq.ops.CSwap, (_QUBIT[0], _QUBIT[1], _QUBIT[2])),
    (fq.ops.RX(0.3), (_QUBIT[0],)),
    (fq.ops.RY(0.3), (_QUBIT[0],)),
    (fq.ops.RZ(0.3), (_QUBIT[0],)),
    (fq.ops.Phase(0.3), (_QUBIT[0],)),
    (fq.ops.CPhase(0.3), (_QUBIT[0], _QUBIT[1])),
    (fq.ops.Shift(1), (_QUTRIT[0],)),
    (fq.ops.Clock(1), (_QUTRIT[0],)),
    (fq.ops.Sum, (_QUTRIT[0], _QUTRIT[1])),
    (fq.ops.SwapLevels(0, 2), (_QUTRIT[0],)),
    (fq.ops.Fourier, (_QUTRIT[0],)),
    (fq.ops.Fourierdg, (_QUTRIT[0],)),
    (fq.ops.SubspaceRX(0.3, (0, 1)), (_QUTRIT[0],)),
    (fq.ops.SubspaceRY(0.3, (0, 1)), (_QUTRIT[0],)),
    (fq.ops.SubspaceRZ(0.3, (0, 1)), (_QUTRIT[0],)),
    (fq.ops.CClock(1), (_QUTRIT[0], _QUTRIT[1])),
]


def _content_code(matrix):
    """Classify through the single classification rule, `_classify_matrix`."""
    d = matrix.shape[0]
    columns = np.empty(d, dtype=np.int64)
    values = np.empty(d, dtype=np.complex128)
    contiguous = np.ascontiguousarray(matrix, dtype=np.complex128)
    return int(_classify_matrix(contiguous, columns, values))


def test_every_key_has_a_spec():
    assert set(_KERNEL_SPECS) == set(BuiltinKernelKey)


def test_kernel_specs_agree_with_content_classification():
    """The declared structure must match what a content scan finds.

    For generic parameters the two must agree exactly; a _DENSE declaration
    on a gate whose special parameter values are more structured is the one
    permitted (and safe) looseness, but at generic parameters a mismatch
    means either a wrong declaration or a missed specialization.
    """
    default_map = default_matrix_implementation_map()
    seen = set()
    for op, targets in _GATE_CASES:
        rule = default_map.implementation_for(type(op))
        key = rule._kernel_key(op, targets=targets)
        seen.add(key)
        matrix = np.asarray(rule(op, targets=targets), dtype=complex)
        assert _KERNEL_SPECS[key] == _content_code(matrix), type(op).__name__
    assert seen == set(BuiltinKernelKey)  # the case table covers every gate


def _counts(simulator_cls, plan, dims, n_clbits, shots, seed, request):
    simulator = simulator_cls()
    simulator.initialize(dims, n_clbits)
    raw = simulator.run(plan, shots, seed, request)
    return list(zip(raw.outcome_keys.tolist(), raw.outcome_counts.tolist()))


def _plan_and_request(program):
    backend = SimulatorBackend()
    plan, _ = backend._lower(program, backend.resolve_layout(program))
    return plan, backend._request_cls(counts=True, statevector=False)


def test_keyed_dispatch_matches_numpy_across_structure_classes():
    # Diagonal (RZ, CZ), permutation (X, CX, Swap), and dense (H, RX) gates
    # in one circuit; fast path, counts identical shot-for-shot.
    program = fq.Program(3, 3)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.RZ(0.4), 0)
    program.add(fq.ops.CX, (0, 1))
    program.add(fq.ops.Swap, (1, 2))
    program.add(fq.ops.CZ, (0, 2))
    program.add(fq.ops.RX(1.1), 2)
    program.add(fq.ops.X, 1)
    program.add_measurement((0, 1, 2), (0, 1, 2))
    plan, request = _plan_and_request(program)

    numpy_counts = _counts(NumpySVSimulator, plan, (2, 2, 2), 3, 300, 11, request)
    numba_counts = _counts(NumbaSVSimulator, plan, (2, 2, 2), 3, 300, 11, request)
    assert numpy_counts == numba_counts


def test_keyed_dispatch_matches_numpy_on_the_dynamic_path():
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    program.add(fq.ops.RZ(0.7), 1)
    program.add_measurement(1, 1)
    plan, request = _plan_and_request(program)

    numpy_counts = _counts(NumpySVSimulator, plan, (2, 2), 2, 20, 13, request)
    numba_counts = _counts(NumbaSVSimulator, plan, (2, 2), 2, 20, 13, request)
    assert numpy_counts == numba_counts


def test_unkeyed_step_content_scan_matches_keyed_dispatch():
    # The same X matrix keyed and un-keyed must produce identical states:
    # the key changes where identification happens, never the numbers.
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    keyed = ApplyMatrixStep(
        matrix=x_matrix, target_indices=(0,), kernel_key=BuiltinKernelKey.X
    )
    unkeyed = ApplyMatrixStep(matrix=x_matrix, target_indices=(0,))
    states = []
    for step in (keyed, unkeyed):
        simulator = NumbaSVSimulator()
        simulator.initialize((2, 2), 0)
        simulator.apply(step)
        states.append(simulator.export_state())
    assert np.array_equal(states[0], states[1])


def test_dense_declaration_stays_correct_at_special_parameters():
    # RX is declared _DENSE; at theta=pi its matrix happens to be a
    # permutation. The declaration must stay numerically correct (the dense
    # kernel handles any matrix), just without the opportunistic speedup.
    program = fq.Program(1)
    program.add(fq.ops.RX(np.pi), 0)
    backend = SimulatorBackend()
    plan, _ = backend._lower(program, backend.resolve_layout(program))
    (step,) = plan
    assert _KERNEL_SPECS[step.kernel_key] == _DENSE

    simulator = NumbaSVSimulator()
    simulator.initialize((2,), 0)
    simulator.apply(step)
    assert np.allclose(simulator.export_state(), [0.0, -1.0j])


def test_fallback_path_matches_the_standard_path():
    # apply(step) (standard, key-aware, cached) and _apply_local (matrix-only
    # fallback used by reset shifts and Kraus branches) must produce the same
    # state through the same resolved kernels.
    matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    standard = NumbaSVSimulator()
    standard.initialize((2, 2), 0)
    standard.apply(
        ApplyMatrixStep(
            matrix=matrix, target_indices=(1,), kernel_key=BuiltinKernelKey.X
        )
    )
    fallback = NumbaSVSimulator()
    fallback.initialize((2, 2), 0)
    fallback._state = fallback._apply_local(fallback.state, matrix, (1,))

    assert np.array_equal(standard.export_state(), fallback.export_state())


def test_structure_cache_pins_steps_and_reuses_resolutions():
    simulator = NumbaSVSimulator()
    simulator.initialize((2,), 0)
    step = ApplyMatrixStep(
        matrix=np.eye(2, dtype=complex),
        target_indices=(0,),
        kernel_key=BuiltinKernelKey.I,
    )
    first = simulator._resolve_structure(step)
    assert simulator._resolve_structure(step) is first  # cached, not re-resolved
    assert simulator._structure_cache[id(step)][0] is step  # pinned identity
