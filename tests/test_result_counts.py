from qnsim.result import (
    build_counts,
    build_counts_from_clbits,
    format_count_key,
)


def test_from_clbits_key_ordering_matches_build_counts():
    # clbit 0 first in the tuple key. snapshot [c0, c1] = [1, 0] -> (1, 0).
    counts = build_counts_from_clbits([[1, 0], [1, 0], [0, 1]], n_clbits=2)
    assert counts == {(1, 0): 2, (0, 1): 1}


def test_from_clbits_agrees_with_build_counts_on_static_case():
    # A single measured qubit q0 -> c0; sampled index 1 means c0 == 1.
    via_indices = build_counts(
        [1, 1, 0], n_clbits=1, measurements=[(0, 0)], system_dims=(2,)
    )
    via_clbits = build_counts_from_clbits([[1], [1], [0]], n_clbits=1)
    assert via_indices == via_clbits == {(1,): 2, (0,): 1}


def test_from_clbits_empty():
    assert build_counts_from_clbits([], n_clbits=2) == {}


def test_build_counts_binary_tuple_keys():
    # 2 qubits measured into clbits 0,1; sampled flat index 0b10 = 2 -> q1=1,q0=0.
    counts = build_counts([2, 2], n_clbits=2, measurements=[(0, 0), (1, 1)], system_dims=(2, 2))
    assert counts == {(0, 1): 2}  # clbit0=0, clbit1=1


def test_build_counts_qutrit_digit_decode():
    # 1 qutrit (subsystem 0) measured into clbit 0. Flat index 2 -> digit 2.
    counts = build_counts([2, 1, 0], n_clbits=1, measurements=[(0, 0)], system_dims=(3,))
    assert counts == {(2,): 1, (1,): 1, (0,): 1}


def test_build_counts_from_clbits_tuple_keys():
    counts = build_counts_from_clbits([(0, 1), (0, 1), (2, 0)], n_clbits=2)
    assert counts == {(0, 1): 2, (2, 0): 1}


def test_format_count_key_single_digit_little_endian():
    # clbit0=0, clbit1=1 -> little-endian string "10".
    assert format_count_key((0, 1), classical_dims=(2, 2)) == "10"


def test_format_count_key_delimited_when_dim_ge_10():
    # clbits (3, 0, 15, 2) with a dim>=10 register -> delimited, little-endian.
    assert format_count_key((3, 0, 15, 2), classical_dims=(4, 4, 16, 4)) == "2,15,0,3"


def test_format_count_key_high_quantum_dim_stays_plain():
    # classical dims are small even though a quantum register was dim 11 upstream;
    # format only sees classical_dims, so the plain string is used.
    assert format_count_key((0, 1), classical_dims=(2, 2)) == "10"
