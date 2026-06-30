import qnsim as qs


def test_backend_run_is_repeatable():
    program = qs.Program(1, 1)
    program.add(qs.ops.X, 0)
    program.add_measurement(0, 0)

    backend = qs.StateVectorBackend(seed=0)
    first = backend.run(program, shots=10).result().get_counts()
    second = backend.run(program, shots=10).result().get_counts()

    # X|0> = |1>, so every shot reads 1 on both runs; the second run must NOT
    # start from the leftover |1> state (which X would flip back to |0>).
    assert first == {"1": 10}
    assert second == {"1": 10}
