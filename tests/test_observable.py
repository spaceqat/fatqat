"""Observable construction: the three input shapes, storage, and validation.

Asserts on the constructed value object only - expectation values are covered
by the kernel and Estimator tests.
"""

import numpy as np
import pytest

import fatqat as fq
from fatqat.observable import Observable

# --- the three constructors agree -------------------------------------------


def test_dense_pairs_labels_and_sparse_agree():
    pairs = Observable([("XI", 1.5)])
    labels = Observable(["XI"], coeffs=[1.5])
    sparse = Observable.from_sparse([("X", (0,), 1.5)], num_qubits=2)

    assert pairs == labels == sparse
    assert pairs.num_qubits == 2


def test_sparse_only_stores_non_identity_factors():
    # A two-body term on 100 qubits stores two factors, not 100.
    obs = Observable.from_sparse([("XY", (3, 7), 1.0)], num_qubits=100)

    ((coeff, factors),) = obs.terms
    assert coeff == 1.0
    assert factors == ((3, "X"), (7, "Y"))
    assert obs.num_qubits == 100


def test_dense_label_uses_public_qubit_order():
    ((_, factors),) = Observable([("XIZ", 1.0)]).terms

    assert factors == ((0, "X"), (2, "Z"))


def test_factor_order_does_not_change_identity():
    # The same term written with qubits in either order compares equal.
    a = Observable.from_sparse([("XY", (3, 7), 1.0)], num_qubits=8)
    b = Observable.from_sparse([("YX", (7, 3), 1.0)], num_qubits=8)

    assert a == b


# --- projectors --------------------------------------------------------------


def test_multi_character_letters_are_not_split():
    # "ONE" is one letter, not O-N-E; "XY" is still two letters.
    ((_, one),) = Observable.from_sparse([("ONE", (5,), 1.0)], num_qubits=8).terms
    ((_, xy),) = Observable.from_sparse([("XY", (0, 1), 1.0)], num_qubits=8).terms

    assert one == ((5, "ONE"),)
    assert xy == ((0, "X"), (1, "Y"))


def test_mixed_letters_via_explicit_sequence():
    obs = Observable.from_sparse([(["ONE", "Z"], (5, 3), 1.0)], num_qubits=8)

    ((_, factors),) = obs.terms
    assert factors == ((3, "Z"), (5, "ONE"))


def test_is_diagonal_distinguishes_projectors_from_pauli_xy():
    assert Observable.from_sparse([("ONE", (0,), 1.0)], num_qubits=2).is_diagonal()
    assert Observable([("ZZ", 1.0)]).is_diagonal()
    assert not Observable([("XZ", 1.0)]).is_diagonal()


# --- special terms (design doc 2.5) -----------------------------------------


def test_identity_term_is_supported():
    # Constant offsets (e.g. nuclear repulsion) must be expressible.
    obs = Observable([("II", 2.5)])

    ((coeff, factors),) = obs.terms
    assert (coeff, factors) == (2.5, ())


def test_repeated_terms_are_kept_separate():
    # No implicit rewriting: two ZZ entries stay two entries.
    obs = Observable([("ZZ", 1.0), ("ZZ", 0.5)])

    assert len(obs) == 2


def test_zero_coefficient_term_is_kept():
    obs = Observable([("ZZ", 0.0)])

    assert len(obs) == 1
    assert obs.terms[0][0] == 0.0


def test_observable_is_reusable_and_hashable():
    obs = Observable([("ZZ", 1.0)])

    assert obs == Observable([("ZZ", 1.0)])
    assert len({obs, Observable([("ZZ", 1.0)])}) == 1


# --- validation --------------------------------------------------------------


def test_complex_coefficient_rejected():
    # Every letter is Hermitian, so Hermiticity reduces to real coefficients.
    with pytest.raises(ValueError, match="Hermitian"):
        Observable([("ZZ", 1j)])


def test_empty_observable_rejected():
    with pytest.raises(ValueError, match="at least one term"):
        Observable([])
    with pytest.raises(ValueError, match="at least one term"):
        Observable.from_sparse([], num_qubits=2)


def test_ragged_labels_rejected():
    with pytest.raises(ValueError, match="same length"):
        Observable([("ZZ", 1.0), ("Z", 1.0)])


def test_num_qubits_disagreeing_with_label_rejected():
    with pytest.raises(ValueError, match="disagrees"):
        Observable([("ZZ", 1.0)], num_qubits=3)


def test_unknown_letter_rejected():
    with pytest.raises(ValueError, match="unknown letter"):
        Observable([("ZQ", 1.0)])
    with pytest.raises(ValueError, match="unknown letter"):
        Observable.from_sparse([("PLUS", (0,), 1.0)], num_qubits=2)


def test_projector_not_reachable_from_dense_label():
    # ZERO/ONE have multi-character names; the dense form is single-character.
    with pytest.raises(ValueError, match="from_sparse"):
        Observable([("ONE", 1.0)], num_qubits=3)


def test_sparse_shape_and_range_validation():
    with pytest.raises(ValueError, match="same length"):
        Observable.from_sparse([("XY", (0,), 1.0)], num_qubits=4)
    with pytest.raises(ValueError, match="out of range"):
        Observable.from_sparse([("X", (9,), 1.0)], num_qubits=4)
    with pytest.raises(ValueError, match="more than once"):
        Observable.from_sparse([("XY", (1, 1), 1.0)], num_qubits=4)
    with pytest.raises(ValueError, match=">= 1"):
        Observable.from_sparse([("X", (0,), 1.0)], num_qubits=0)


def test_mismatched_label_and_coeff_counts_rejected():
    with pytest.raises(ValueError, match="coefficient"):
        Observable(["ZZ", "XX"], coeffs=[1.0])


def test_wrong_input_shapes_rejected():
    with pytest.raises(TypeError, match="pairs"):
        Observable(["ZZ"])  # bare labels without coeffs=
    with pytest.raises(TypeError, match="label strings"):
        Observable([("ZZ", 1.0)], coeffs=[1.0])


def test_exported_at_top_level():
    assert fq.Observable is Observable


# --- numpy integers as qubit indices -----------------------------------------


def test_numpy_integers_are_accepted_as_qubits():
    # Building a many-body observable from numpy is the natural way to do it -
    # rng.choice for a lattice sample, arange for a chain - and those yield
    # np.int64 rather than int.
    rng = np.random.default_rng(0)
    qubits = tuple(rng.choice(8, 2, replace=False))
    assert all(isinstance(q, np.integer) for q in qubits)

    observable = Observable.from_sparse([("ZZ", qubits, 1.0)], num_qubits=np.int64(8))

    assert observable.num_qubits == 8
    assert isinstance(observable.num_qubits, int)


def test_numpy_qubits_are_narrowed_to_python_ints():
    # Stored as plain ints so terms hash and repr the same however they were
    # built, and so an equal term compares equal across construction routes.
    from_numpy = Observable.from_sparse(
        [("ZZ", (np.int64(1), np.int32(0)), 1.5)], num_qubits=2
    )
    from_python = Observable.from_sparse([("ZZ", (1, 0), 1.5)], num_qubits=2)

    assert all(
        type(qubit) is int for _, factors in from_numpy.terms for qubit, _ in factors
    )
    assert from_numpy == from_python
    assert hash(from_numpy) == hash(from_python)


@pytest.mark.parametrize(
    "bad", [True, False, np.bool_(True), 1.5, np.float64(1.0), "0", None]
)
def test_non_integer_qubits_still_rejected(bad):
    # Booleans in particular: bool subclasses int and np.bool_ sits beside
    # np.integer, so True would otherwise pass silently as qubit 1.
    with pytest.raises(TypeError, match="must be an int"):
        Observable.from_sparse([("Z", (bad,), 1.0)], num_qubits=4)


@pytest.mark.parametrize("bad", [True, np.bool_(True), 2.0, "4"])
def test_non_integer_num_qubits_still_rejected(bad):
    with pytest.raises(TypeError, match="must be an int"):
        Observable.from_sparse([("Z", (0,), 1.0)], num_qubits=bad)
