"""Tests the top-level fatqat public API surface."""

import fatqat as fq


def test_waveform_and_pulse_authoring_names_are_namespaced():
    from fatqat.emulator import ControlChannel, PulseControl
    from fatqat.operations import PulseOperation
    from fatqat.waveforms import SampledWaveform

    assert fq.emulator.ControlChannel is ControlChannel
    assert fq.emulator.PulseControl is PulseControl
    assert fq.waveforms.SampledWaveform is SampledWaveform
    assert fq.ops.PulseOperation is PulseOperation
    assert not hasattr(fq, "ControlChannel")
    assert not hasattr(fq, "PulseControl")
    assert not hasattr(fq, "PulseOperation")
    assert not hasattr(fq, "SampledWaveform")


def test_atom_3level_arrangement_is_the_top_level_data_value():
    from fatqat.atom_arrangement import AtomArrangement
    from fatqat.emulator.atom_3level import (
        Atom3LevelCalibration,
        Atom3LevelModel,
    )

    assert fq.AtomArrangement is AtomArrangement
    assert not hasattr(fq, "Atom3LevelModel")
    assert not hasattr(fq, "Atom3LevelEmulator")
    assert hasattr(fq.emulator, "Atom3LevelEmulator")
    assert fq.emulator.Atom3LevelModel is Atom3LevelModel
    assert fq.emulator.Atom3LevelCalibration is Atom3LevelCalibration
    assert not hasattr(fq.emulator, "load_atom_physics_model")
    assert not hasattr(fq.emulator, "load_atom_calibration_spec")
    assert not hasattr(fq.emulator, "default_digital_atom_gate_implementation_map")
    assert not hasattr(fq.emulator, "SampledPulseTemplate")


def test_atom_2level_model_values_are_exported_only_from_emulator_namespaces():
    from fatqat.emulator.atom_2level import (
        Atom2LevelEmulator,
        Atom2LevelModel,
        GridInteractionPolicy,
    )

    assert fq.emulator.Atom2LevelEmulator is Atom2LevelEmulator
    assert fq.emulator.Atom2LevelModel is Atom2LevelModel
    assert fq.emulator.GridInteractionPolicy is GridInteractionPolicy
    assert not hasattr(fq.emulator, "Channel" + "Description")
    assert not hasattr(fq.emulator, "ControlComponent" + "Description")
    assert not hasattr(fq.emulator, "load_" + "analog_atom_" + "physics_model")
    for name in (
        "Atom2LevelEmulator",
        "Atom2LevelModel",
        "GridInteractionPolicy",
    ):
        assert not hasattr(fq, name)


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


def test_transmon_emulator_values_are_namespaced_under_emulator():
    from fatqat.emulator.superconducting import (
        TransmonCalibration,
        TransmonEmulator,
        TransmonModel,
        default_transmon_gate_implementation_map,
    )

    assert fq.emulator.TransmonEmulator is TransmonEmulator
    assert fq.emulator.TransmonModel is TransmonModel
    assert fq.emulator.TransmonCalibration is TransmonCalibration
    assert (
        fq.emulator.default_transmon_gate_implementation_map
        is default_transmon_gate_implementation_map
    )
    for name in (
        "TransmonEmulator",
        "TransmonModel",
        "TransmonCalibration",
    ):
        assert not hasattr(fq, name)


def test_resultconfig_not_exported_from_top_level():
    assert not hasattr(fq, "ResultConfig")


def test_error_classes_only_under_errors_namespace():
    from fatqat.errors import (
        FatqatError,
        BackendValidationError,
        MatrixImplementationError,
        UnsupportedOperationError,
        NoMeasurementWarning,
    )

    assert fq.errors.FatqatError is FatqatError
    assert fq.errors.BackendValidationError is BackendValidationError
    assert fq.errors.MatrixImplementationError is MatrixImplementationError
    assert fq.errors.UnsupportedOperationError is UnsupportedOperationError
    assert fq.errors.NoMeasurementWarning is NoMeasurementWarning
    assert not hasattr(fq, "FatqatError")


def test_program_measure_all_is_public_instance_method():
    program = fq.Program(1, 1)
    program.measure_all()

    assert len(program.operations) == 1
