import qnsim as qs


def test_minimal_workflow_from_spec():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    backend = qs.StateVectorBackend(seed=2024)
    job = backend.run(program, shots=1000, result_config=qs.ResultConfig(counts=True))
    result = job.result()
    counts = result.get_counts()

    assert sum(counts.values()) == 1000
    assert set(counts) <= {"00", "01"}
    # roughly balanced between the two reachable outcomes
    assert all(150 < v < 850 for v in counts.values())


def test_backends_module_alias():
    assert qs.backends.StateVectorBackend is qs.StateVectorBackend


def test_error_and_warning_classes_exposed():
    assert issubclass(qs.UnsupportedOperationError, qs.BackendValidationError)
    assert issubclass(qs.BackendValidationError, qs.QnsimError)
    assert issubclass(qs.NoMeasurementWarning, UserWarning)
