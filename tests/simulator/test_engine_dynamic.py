import numpy as np

from fatqat.simulator._engine.np import NumpySVEngine
from fatqat._backends.steps import ApplyMatrixStep

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def test_measure_qubit_deterministic_one():
    eng = NumpySVEngine()
    eng.initialize((2,))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # |1>
    bit = eng.measure_subsystem(0, np.random.default_rng(0))
    assert bit == 1
    assert np.allclose(eng.export_state(), np.array([0, 1], dtype=complex))


def test_measure_qubit_collapses_and_is_repeatable():
    eng = NumpySVEngine()
    eng.initialize((2,))
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))  # (|0>+|1>)/sqrt2
    first = eng.measure_subsystem(0, np.random.default_rng(0))
    # after collapse the state is a basis state; re-measuring returns the same bit
    second = eng.measure_subsystem(0, np.random.default_rng(123))
    assert first == second


def test_reset_qubit_from_one_returns_zero():
    eng = NumpySVEngine()
    eng.initialize((2,))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # |1>
    eng.reset_subsystem(0, np.random.default_rng(0))
    assert np.allclose(eng.export_state(), np.array([1, 0], dtype=complex))


def test_reset_qubit_on_entangled_pair_conditions_the_partner():
    # Bell pair (|00>+|11>)/sqrt2; reset qubit 0 -> partner is |0> or |1>, 50/50.
    outcomes = []
    for s in range(200):
        eng = NumpySVEngine()
        eng.initialize((2, 2))
        eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
        cx = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
        eng.apply(ApplyMatrixStep(matrix=cx, target_indices=(0, 1)))
        eng.reset_subsystem(0, np.random.default_rng(s))
        st = eng.export_state()
        # qubit 0 must be |0>: only indices with bit0 == 0 are allowed.
        # Amplitude is on index 0 (partner 0) or 2 (partner 1).
        nz = np.flatnonzero(np.round(np.abs(st), 6))
        assert nz.size == 1 and nz[0] in (0, 2)
        outcomes.append(int(nz[0] == 2))  # partner==1
    frac = sum(outcomes) / len(outcomes)
    assert 0.35 < frac < 0.65


def test_measure_subsystems_returns_bits_in_requested_order():
    eng = NumpySVEngine()
    eng.initialize((2, 2, 2))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(2,)))

    bits = eng.measure_subsystems((2, 0), np.random.default_rng(0))

    assert bits == (1, 1)
    assert np.allclose(eng.export_state()[0b101], 1.0)


def test_measure_subsystems_consumes_one_rng_draw_for_grouped_event():
    eng_grouped = NumpySVEngine()
    eng_grouped.initialize((2, 2))
    eng_grouped.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
    eng_grouped.apply(
        ApplyMatrixStep(
            matrix=np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=complex,
            ),
            target_indices=(0, 1),
        )
    )
    rng_grouped = np.random.default_rng(11)
    eng_grouped.measure_subsystems((0, 1), rng_grouped)
    after_grouped = rng_grouped.random()

    rng_one_draw = np.random.default_rng(11)
    rng_one_draw.choice(4, p=np.array([0.5, 0.0, 0.0, 0.5]))
    after_one_draw = rng_one_draw.random()

    assert after_grouped == after_one_draw


def test_reset_subsystems_resets_all_targets_with_one_grouped_collapse():
    eng = NumpySVEngine()
    eng.initialize((2, 2))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(1,)))

    eng.reset_subsystems((0, 1), np.random.default_rng(0))

    assert np.allclose(eng.export_state(), np.array([1, 0, 0, 0], dtype=complex))


def test_single_qubit_wrappers_delegate_to_grouped_methods():
    eng = NumpySVEngine()
    eng.initialize((2,))
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))

    bit = eng.measure_subsystem(0, np.random.default_rng(0))

    assert bit == 1
