"""Tests a minimal public qnsim workflow from program construction to counts."""

import qnsim as qs


def test_minimal_workflow_from_spec():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    backend = qs.StateVectorBackend()
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

    result = qs.StateVectorBackend(
        options={"max_workers": 2, "parallel_backend": "multiprocessing"}
    ).run(
        program,
        shots=12,
        seed=2026,
        result_config={"counts": True},
    ).result()

    assert result.get_counts() == {"00": 12}
