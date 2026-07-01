from qnsim.result import build_counts, build_counts_from_clbits


def test_from_clbits_key_ordering_matches_build_counts():
    # clbit 0 rightmost. snapshot [c0, c1] = [1, 0] -> "01".
    counts = build_counts_from_clbits([[1, 0], [1, 0], [0, 1]], n_clbits=2)
    assert counts == {"01": 2, "10": 1}


def test_from_clbits_agrees_with_build_counts_on_static_case():
    # A single measured qubit q0 -> c0; sampled index 1 means c0 == 1.
    via_indices = build_counts([1, 1, 0], n_clbits=1, measurements=[(0, 0)])
    via_clbits = build_counts_from_clbits([[1], [1], [0]], n_clbits=1)
    assert via_indices == via_clbits == {"1": 2, "0": 1}


def test_from_clbits_empty():
    assert build_counts_from_clbits([], n_clbits=2) == {}
