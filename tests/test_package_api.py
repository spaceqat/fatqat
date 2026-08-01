"""Tests the top-level fatqat public API surface."""

import fatqat as fq


def test_top_level_frontend_surface():
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CZ, (0, 1))
    program.add(fq.ops.RX(0.1), 0)
    program.measure(0, 0)
    program.measure(1, 1)

    assert len(program.operations) == 5
    assert program.operations[0].operation.name == "H"
    assert isinstance(program.operations[3], fq.Measurement)


def test_register_types_exposed():
    qr = fq.QuantumRegister(2, name="q")
    assert isinstance(qr[0], fq.RegisterRef)


def test_simulator_only_under_simulator_namespace():
    from fatqat.simulator import Simulator

    assert fq.simulator.Simulator is Simulator
    assert not hasattr(fq, "Simulator")


def test_constrained_targets_exported_under_simulator_namespace():
    from fatqat.simulator import SCQubitGoogleSimulator, SCQubitIBMSimulator

    assert fq.simulator.SCQubitGoogleSimulator is SCQubitGoogleSimulator
    assert fq.simulator.SCQubitIBMSimulator is SCQubitIBMSimulator
    assert not hasattr(fq, "SCQubitGoogleSimulator")
    assert not hasattr(fq, "SCQubitIBMSimulator")


def test_resultconfig_not_exported_from_top_level():
    assert not hasattr(fq, "ResultConfig")


def test_error_classes_only_under_errors_namespace():
    from fatqat.errors import (
        FatqcatError,
        BackendValidationError,
        MatrixImplementationError,
        UnsupportedOperationError,
        NoMeasurementWarning,
    )

    assert fq.errors.FatqcatError is FatqcatError
    assert fq.errors.BackendValidationError is BackendValidationError
    assert fq.errors.MatrixImplementationError is MatrixImplementationError
    assert fq.errors.UnsupportedOperationError is UnsupportedOperationError
    assert fq.errors.NoMeasurementWarning is NoMeasurementWarning
    assert not hasattr(fq, "FatqcatError")


def test_program_measure_all_is_public_instance_method():
    program = fq.Program(1, 1)
    program.measure_all()

    assert len(program.operations) == 1
