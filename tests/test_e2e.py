"""Tests a minimal public qnsim workflow from program construction to counts."""

import qnsim as qs


def test_minimal_workflow_from_spec():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    backend = qs.backends.StateVectorBackend()
    job = backend.run(program, shots=1000, seed=2024, result_config={"counts": True})
    result = job.result()
    counts = result.get_counts()

    assert sum(counts.values()) == 1000
    assert set(counts) <= {"00", "01"}
    # roughly balanced between the two reachable outcomes
    assert all(150 < v < 850 for v in counts.values())


def test_phase3_grouped_measure_reset_and_parallel_counts_workflow():
    program = qs.Program(2, 2)
    program.add(qs.ops.X, 0)
    program.add(qs.ops.X, 1)
    program.add_measurement((0, 1), (0, 1))
    program.add(qs.ops.Reset, (0, 1))
    program.measure_all()

    result = qs.backends.StateVectorBackend(
        options={"max_workers": 2, "parallel_mode": "multiprocessing"}
    ).run(
        program,
        shots=12,
        seed=2026,
        result_config={"counts": True},
    ).result()

    assert result.get_counts() == {"00": 12}


def test_heterogeneous_qutrit_qubit_program():
    qt = qs.QuantumRegister(1, dim=3)
    qb = qs.QuantumRegister(1, dim=2)
    ct = qs.ClassicalRegister(1, dim=3)
    cb = qs.ClassicalRegister(1, dim=2)
    program = qs.Program([qt, qb], [ct, cb])
    program.add(qs.ops.Shift(1), qt[0])  # qutrit |0> -> |1>
    program.add(qs.ops.X, qb[0])         # qubit  |0> -> |1>
    program.add_measurement(qt[0], ct[0])
    program.add_measurement(qb[0], cb[0])
    result = qs.backends.StateVectorBackend().run(program, shots=16).result()
    assert result.get_counts_as_tuples() == {(1, 1): 16}


def test_sum_across_mismatched_dims_fails_at_lowering():
    import pytest
    from qnsim.errors import MatrixImplementationError

    qt = qs.QuantumRegister(1, dim=3)
    qb = qs.QuantumRegister(1, dim=2)
    program = qs.Program([qt, qb])
    program.add(qs.ops.Sum, (qt[0], qb[0]))  # frontend does not raise
    with pytest.raises(MatrixImplementationError):
        qs.backends.StateVectorBackend().run(
            program, result_config={"counts": False, "statevector": True}
        ).result()


def test_sum_entangles_two_qutrits():
    qreg = qs.QuantumRegister(2, dim=3)
    creg = qs.ClassicalRegister(2, dim=3)
    program = qs.Program([qreg], [creg])
    program.add(qs.ops.Shift(2), 0)      # control qutrit -> |2>
    program.add_measurement(0, 0)        # clbit0 = 2 (mid-circuit; deterministic)
    # Condition genuinely fires (clbit0 == 2, a value only reachable by a
    # qudit, not just 0/1), proving condition literals are compared for exact
    # equality rather than truthiness for dim > 2.
    program.add(qs.ops.Sum, (0, 1), condition=(creg[0], 2))  # target -> (2+0)%3 = 2
    program.add_measurement(1, 1)
    result = qs.backends.StateVectorBackend().run(program, shots=32).result()
    assert result.get_counts_as_tuples() == {(2, 2): 32}


def test_fast_and_dynamic_counts_match_for_qutrit():
    def build(force_dynamic):
        qreg = qs.QuantumRegister(1, dim=3)
        creg = qs.ClassicalRegister(1, dim=3)
        p = qs.Program([qreg], [creg])
        p.add(qs.ops.Shift(2), 0)          # deterministic |0> -> |2>
        p.add_measurement(0, 0)
        if force_dynamic:
            # Inert no-op: Shift(0) is the identity, and its condition can
            # never be satisfied (the measured clbit is always 2), so this
            # cannot change the outcome distribution. Its mere presence
            # (a condition) forces backends.py's is_dynamic classification,
            # letting the dynamic path be compared against the fast path for
            # the identical program shape and seed.
            p.add(qs.ops.Shift(0), 0, condition=(p.creg[0][0], 0))
        return p

    fast_counts = (
        qs.backends.StateVectorBackend().run(build(False), shots=8, seed=7).result().get_counts_as_tuples()
    )
    dyn_counts = (
        qs.backends.StateVectorBackend().run(build(True), shots=8, seed=7).result().get_counts_as_tuples()
    )
    assert fast_counts == dyn_counts == {(2,): 8}
