"""Tests the top-level fatqcat public API surface."""

import fatqcat as fqc


def test_top_level_frontend_surface():
    program = fqc.Program(2, 2)
    program.add(fqc.ops.H, 0)
    program.add(fqc.ops.CZ, (0, 1))
    program.add(fqc.ops.RX(0.1), 0)
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    assert len(program.operations) == 5
    assert program.operations[0].operation.name == "H"
    assert isinstance(program.operations[3], fqc.Measurement)


def test_register_types_exposed():
    qr = fqc.QuantumRegister(2, name="q")
    assert isinstance(qr[0], fqc.RegisterRef)


def test_statevector_backend_only_under_backends_namespace():
    from fatqcat.backends import StateVectorBackend

    assert fqc.backends.StateVectorBackend is StateVectorBackend
    assert not hasattr(fqc, "StateVectorBackend")


def test_resultconfig_not_exported_from_top_level():
    assert not hasattr(fqc, "ResultConfig")


def test_error_classes_only_under_errors_namespace():
    from fatqcat.errors import (
        FatqcatError,
        BackendValidationError,
        MatrixImplementationError,
        UnsupportedOperationError,
        NoMeasurementWarning,
    )
    assert fqc.errors.FatqcatError is FatqcatError
    assert fqc.errors.BackendValidationError is BackendValidationError
    assert fqc.errors.MatrixImplementationError is MatrixImplementationError
    assert fqc.errors.UnsupportedOperationError is UnsupportedOperationError
    assert fqc.errors.NoMeasurementWarning is NoMeasurementWarning
    assert not hasattr(fqc, "FatqcatError")


def test_program_measure_all_is_public_instance_method():
    program = fqc.Program(1, 1)
    program.measure_all()

    assert len(program.operations) == 1


