"""Tests the top-level fatqat public API surface."""

import fatqat as fc


def test_top_level_frontend_surface():
    program = fc.Program(2, 2)
    program.add(fc.ops.H, 0)
    program.add(fc.ops.CZ, (0, 1))
    program.add(fc.ops.RX(0.1), 0)
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    assert len(program.operations) == 5
    assert program.operations[0].operation.name == "H"
    assert isinstance(program.operations[3], fc.Measurement)


def test_register_types_exposed():
    qr = fc.QuantumRegister(2, name="q")
    assert isinstance(qr[0], fc.RegisterRef)


def test_statevector_backend_only_under_backends_namespace():
    from fatqat.backends import StateVectorBackend

    assert fc.backends.StateVectorBackend is StateVectorBackend
    assert not hasattr(fc, "StateVectorBackend")


def test_resultconfig_not_exported_from_top_level():
    assert not hasattr(fc, "ResultConfig")


def test_error_classes_only_under_errors_namespace():
    from fatqat.errors import (
        FatqcatError,
        BackendValidationError,
        MatrixImplementationError,
        UnsupportedOperationError,
        NoMeasurementWarning,
    )
    assert fc.errors.FatqcatError is FatqcatError
    assert fc.errors.BackendValidationError is BackendValidationError
    assert fc.errors.MatrixImplementationError is MatrixImplementationError
    assert fc.errors.UnsupportedOperationError is UnsupportedOperationError
    assert fc.errors.NoMeasurementWarning is NoMeasurementWarning
    assert not hasattr(fc, "FatqcatError")


def test_program_measure_all_is_public_instance_method():
    program = fc.Program(1, 1)
    program.measure_all()

    assert len(program.operations) == 1


