import json

import numpy as np

from fatqat.result import (
    counts_dict_from_arrays,
    decode_indices_to_clbit_rows,
    format_count_key,
    reduce_to_counts,
)


def test_reduce_to_counts_key_ordering_from_clbit_rows():
    rows = np.array([[1, 0], [1, 0], [0, 1]], dtype=int)
    keys, counts = reduce_to_counts(rows)
    assert counts_dict_from_arrays(keys, counts) == {(1, 0): 2, (0, 1): 1}


def test_reduce_to_counts_agrees_with_decoded_static_case():
    via_indices = decode_indices_to_clbit_rows(
        [1, 1, 0], measurements=[(0, 0)], system_dims=(2,), n_clbits=1
    )
    keys_from_indices, counts_from_indices = reduce_to_counts(via_indices)
    keys_from_clbits, counts_from_clbits = reduce_to_counts(
        np.array([[1], [1], [0]], dtype=int)
    )
    assert counts_dict_from_arrays(keys_from_indices, counts_from_indices) == {
        (1,): 2,
        (0,): 1,
    }
    assert counts_dict_from_arrays(keys_from_clbits, counts_from_clbits) == {
        (1,): 2,
        (0,): 1,
    }


def test_reduce_to_counts_empty_rows():
    keys, counts = reduce_to_counts(np.empty((0, 2), dtype=int))
    assert keys.shape == (0, 2)
    assert counts.shape == (0,)
    assert counts_dict_from_arrays(keys, counts) == {}


def test_decode_indices_to_clbit_rows_binary_tuple_keys():
    # 2 qubits measured into clbits 0,1; sampled flat index 0b10 = 2 -> q1=1,q0=0.
    rows = decode_indices_to_clbit_rows(
        [2, 2], measurements=[(0, 0), (1, 1)], system_dims=(2, 2), n_clbits=2
    )
    assert np.array_equal(rows, np.array([[0, 1], [0, 1]], dtype=int))


def test_decode_indices_to_clbit_rows_qutrit_digit_decode():
    # 1 qutrit (subsystem 0) measured into clbit 0. Flat index 2 -> digit 2.
    rows = decode_indices_to_clbit_rows(
        [2, 1, 0], measurements=[(0, 0)], system_dims=(3,), n_clbits=1
    )
    assert np.array_equal(rows, np.array([[2], [1], [0]], dtype=int))


def test_reduce_to_counts_tuple_keys():
    rows = np.array([(0, 1), (0, 1), (2, 0)], dtype=int)
    keys, counts = reduce_to_counts(rows)
    assert counts_dict_from_arrays(keys, counts) == {(0, 1): 2, (2, 0): 1}


def test_counts_dict_from_arrays_returns_python_ints():
    keys = np.array([[1, 0]], dtype=np.int64)
    counts = np.array([3], dtype=np.int64)
    result = counts_dict_from_arrays(keys, counts)
    key = next(iter(result))
    value = result[key]
    assert key == (1, 0)
    assert all(type(part) is int for part in key)
    assert type(value) is int
    json.dumps({"counts": [(key, value)]})


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


def test_delimited_key_threshold_end_to_end():
    import fatqat as fq

    # Two classical slots so plain concatenation ("310") and the delimited
    # little-endian form ("3,10") are visibly different strings -- a single
    # dim>=10 clbit can't distinguish the two formats (e.g. "10" is the same
    # string either way), so this must use >= 2 slots to actually pin the
    # delimited branch.
    qreg = fq.QuantumRegister(2, dim=11)
    creg = fq.ClassicalRegister(2, dim=11)
    program = fq.Program([qreg], [creg])
    program.add(fq.ops.Shift(10), qreg[0])  # |0> -> |10>
    program.add(fq.ops.Shift(3), qreg[1])   # |0> -> |3>
    program.add_measurement((qreg[0], qreg[1]), (creg[0], creg[1]))
    result = fq.backends.StateVectorBackend().run(program, shots=4).result()
    # clbit0=10, clbit1=3; little-endian (highest clbit first): "3,10".
    assert result.get_counts() == {"3,10": 4}
    assert result.get_counts_as_tuples() == {(10, 3): 4}


def test_high_quantum_dim_low_classical_stays_plain():
    import fatqat as fq

    # dim-11 quantum register measured into... impossible (dims must match);
    # instead: a dim-11 quantum register left UNMEASURED, low-dim classical slots.
    qbig = fq.QuantumRegister(1, dim=11)
    qb = fq.QuantumRegister(1, dim=2)
    cb = fq.ClassicalRegister(1, dim=2)
    program = fq.Program([qbig, qb], [cb])
    program.add(fq.ops.X, qb[0])
    program.add_measurement(qb[0], cb[0])
    result = fq.backends.StateVectorBackend().run(program, shots=4).result()
    assert result.get_counts() == {"1": 4}  # plain string; classical dims are all <= 9
