"""Tests the top-level qnsim public API surface."""

import qnsim as qs


def test_top_level_frontend_surface():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add(qs.ops.RX(0.1), 0)
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    assert len(program.operations) == 5
    assert program.operations[0].operation.name == "H"
    assert isinstance(program.operations[3], qs.Measurement)


def test_register_types_exposed():
    qr = qs.QuantumRegister(2, name="q")
    assert isinstance(qr[0], qs.RegisterRef)


def test_statevector_backend_only_under_backends_namespace():
    from qnsim.backends import StateVectorBackend

    assert qs.backends.StateVectorBackend is StateVectorBackend
    assert not hasattr(qs, "StateVectorBackend")


def test_resultconfig_not_exported_from_top_level():
    assert not hasattr(qs, "ResultConfig")


def test_error_classes_only_under_errors_namespace():
    from qnsim.errors import (
        QnsimError,
        BackendValidationError,
        MatrixImplementationError,
        UnsupportedOperationError,
        NoMeasurementWarning,
    )
    assert qs.errors.QnsimError is QnsimError
    assert qs.errors.BackendValidationError is BackendValidationError
    assert qs.errors.MatrixImplementationError is MatrixImplementationError
    assert qs.errors.UnsupportedOperationError is UnsupportedOperationError
    assert qs.errors.NoMeasurementWarning is NoMeasurementWarning
    assert not hasattr(qs, "QnsimError")


def test_sumgate_class_not_in_ops_public_surface():
    assert "SumGate" not in qs.ops.__all__
    assert isinstance(qs.ops.Sum, qs.ops.SumGate)


def test_program_measure_all_is_public_instance_method():
    program = qs.Program(1, 1)
    program.measure_all()

    assert len(program.operations) == 1
