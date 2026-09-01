"""Public release-surface tests: export identity and private-type exclusion.

Strictly about what `fatqat.emulator` re-exports. The documented custom-CZ
workflow that exercises those exports end to end lives in
`test_pulse_custom_implementation_workflow.py`.
"""

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, get_type_hints

import pytest

import fatqat as fq
from fatqat.emulator import (
    Atom2LevelEmulator,
    Atom2LevelModel,
    PhaseShift,
    PhaseSwap,
    PulseControl,
    PulseDefinition,
    PulseImplementationMap,
    ControlChannel,
    TransmonEmulator,
    TransmonCalibration,
    TransmonModel,
    default_transmon_calibration,
    default_transmon_gate_implementation_map,
    generate_transmon_grid_documents,
)
from fatqat.emulator._core.pulse import PhaseShift as _PhaseShift
from fatqat.emulator._core.pulse import PhaseSwap as _PhaseSwap
from fatqat.emulator._core.pulse import PulseDefinition as _PulseDefinition
from fatqat.emulator._core.pulse import (
    PulseImplementationMap as _PulseImplementationMap,
)
from fatqat._pulse_values import ControlChannel as _ControlChannel
from fatqat._pulse_values import PulseControl as _PulseControl
from fatqat.emulator._core.backend import _PulseBackend
from fatqat.job import Job
from fatqat.resource_layout import DeviceOperand


def test_sc_pulse_factories_are_public_without_exposing_execution_types():
    assert fq.emulator.TransmonEmulator is TransmonEmulator
    assert not hasattr(fq.emulator, "Emulator")
    assert not hasattr(fq.emulator, "waveform")
    assert fq.emulator.TransmonModel is TransmonModel
    assert fq.emulator.TransmonCalibration is TransmonCalibration
    assert (
        fq.emulator.generate_transmon_grid_documents is generate_transmon_grid_documents
    )
    assert fq.emulator.default_transmon_calibration is default_transmon_calibration
    assert not hasattr(fq.emulator, "_TransmonQutipAdapter")
    assert not hasattr(fq.emulator, "PulseEngine")
    assert not hasattr(fq.emulator, "PulseBlock")


def test_family_and_aggregate_exports_are_exact():
    from fatqat.emulator import atom_2level, superconducting

    assert tuple(fq.emulator.__all__) == (
        "TransmonEmulator",
        "Atom2LevelEmulator",
        "TransmonModel",
        "TransmonCalibration",
        "PulseDefinition",
        "ControlChannel",
        "PulseControl",
        "SampledWaveform",
        "PhaseShift",
        "PhaseSwap",
        "PulseImplementationMap",
        "default_transmon_gate_implementation_map",
        "default_transmon_calibration",
        "generate_transmon_grid_documents",
        "Atom2LevelModel",
        "AtomArrangement",
        "available_model_documents",
        "load_model_document",
    )
    assert tuple(atom_2level.__all__) == (
        "Atom2LevelEmulator",
        "Atom2LevelModel",
    )
    assert tuple(superconducting.__all__) == (
        "TransmonEmulator",
        "TransmonCalibration",
        "TransmonModel",
        "angular_rate_from_ghz",
        "default_transmon_calibration",
        "default_transmon_gate_implementation_map",
        "generate_transmon_grid_documents",
    )


def test_transmon_public_names_have_exact_modules_and_constructor_signature():
    assert TransmonModel.__module__ == "fatqat.emulator.superconducting.model"
    assert (
        TransmonCalibration.__module__ == "fatqat.emulator.superconducting.calibration"
    )
    assert TransmonEmulator.__module__ == "fatqat.emulator.superconducting.backend"
    assert (
        generate_transmon_grid_documents.__module__
        == "fatqat.emulator.superconducting.grid_reference"
    )

    generator_parameters = inspect.signature(
        generate_transmon_grid_documents
    ).parameters
    assert tuple(generator_parameters) == (
        "shape",
        "frequency_groups_ghz",
        "frequency_std_ghz",
        "anharmonicity_ghz",
        "seed",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in generator_parameters.values()
    )
    assert generator_parameters["frequency_std_ghz"].default == 0.010
    assert generator_parameters["anharmonicity_ghz"].default == -0.22
    assert generator_parameters["seed"].default == 0
    assert (
        get_type_hints(generate_transmon_grid_documents)["return"]
        == tuple[dict[str, Any], dict[str, Any]]
    )

    parameters = inspect.signature(TransmonEmulator).parameters
    assert tuple(parameters) == (
        "model",
        "method",
        "noise",
        "gate_implementation_map",
    )
    assert parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in (
            "method",
            "noise",
            "gate_implementation_map",
        )
    )
    assert parameters["method"].default == "statevector"
    assert parameters["noise"].default is None
    assert parameters["gate_implementation_map"].default is None
    assert "pulse_" + "implementation_map" not in parameters
    assert not hasattr(TransmonEmulator, "propagator")
    assert not hasattr(TransmonEmulator, "apply_final_frame")


def test_removed_transmon_surface_has_no_compatibility_aliases():
    from fatqat.emulator import superconducting

    removed_names = (
        "SCQubit" + "Emulator",
        "SCTransmon" + "Model",
        "SCTransmon" + "Calibration",
        "TransmonGridReference",
        "generate_transmon_grid_reference",
    )
    removed_helper = (
        "default_" + "superconducting_" + "pulse_" + "implementation_" + "map"
    )
    for owner in (fq.emulator, superconducting):
        for name in removed_names:
            assert not hasattr(owner, name)
            assert name not in owner.__all__
        assert not hasattr(owner, removed_helper)
        assert removed_helper not in owner.__all__


def test_document_identities_and_normalized_records_are_not_public():
    from fatqat.emulator import superconducting

    assert fq.emulator.Atom2LevelModel is Atom2LevelModel
    for name in (
        "FormatIdentity",
        "ModelIdentity",
        "CalibrationIdentity",
        "Transmon",
        "Coupling",
    ):
        assert not hasattr(fq.emulator, name)
        assert name not in fq.emulator.__all__
        assert not hasattr(superconducting, name)
        assert name not in superconducting.__all__

    for name in (
        "TransmonModel",
        "TransmonCalibration",
        "generate_transmon_grid_documents",
        "Atom3LevelModel",
        "Atom3LevelCalibration",
        "Atom2LevelModel",
    ):
        assert not hasattr(fq, name)


def test_removed_construction_surfaces_have_no_compatibility_aliases():
    removed_names = (
        "load_physics_model",
        "load_calibration_spec",
        "load_atom_physics_model",
        "load_atom_calibration_spec",
        "SCTransmonExchangeBuilder",
        "CalibrationSpec",
        "Physics" + "ModelSpec",
        "Physics" + "ModelBuilderRegistry",
        "ModelKey",
        "BuilderIdentity",
    )
    for name in (*removed_names, "load_" + "analog_atom_" + "physics_model"):
        assert not hasattr(fq.emulator, name)
        assert name not in fq.emulator.__all__


def test_three_level_atom_family_is_absent_from_the_public_surface():
    atom3_names = (
        "Atom3LevelEmulator",
        "Atom3LevelModel",
        "Atom3LevelCalibration",
        "default_atom_3level_calibration",
        "default_atom_3level_gate_implementation_map",
    )
    legacy_names = (
        "Atom" + "Emulator",
        "Digital" + "Atom" + "Model",
        "Digital" + "Atom" + "Calibration",
        "Atom3Level" + "Frame" + "Ref",
    )
    for name in (*atom3_names, *legacy_names):
        assert not hasattr(fq.emulator, name)
        assert name not in fq.emulator.__all__
    assert not hasattr(fq.emulator, "atom_3level")
    assert importlib.util.find_spec("fatqat.emulator.atom_3level") is None
    assert importlib.util.find_spec("fatqat.emulator." + "atom") is None


def test_atom_2level_family_exports_final_public_values_only():
    from fatqat.emulator import atom_2level

    assert fq.emulator.Atom2LevelEmulator is Atom2LevelEmulator
    assert fq.emulator.Atom2LevelModel is Atom2LevelModel
    assert not hasattr(fq.emulator, "GridInteractionPolicy")
    assert not hasattr(atom_2level, "GridInteractionPolicy")
    assert Atom2LevelEmulator.__module__ == "fatqat.emulator.atom_2level.backend"
    assert Atom2LevelModel.__module__ == "fatqat.emulator.atom_2level.model"
    for private_name in (
        "_Prepared" + "Atom2LevelRuntime",
        "_Atom2LevelQutipAdapter",
        "_Atom2LevelPulseBackend",
    ):
        assert not hasattr(fq.emulator, private_name)
        assert not hasattr(atom_2level, private_name)


def test_removed_two_level_atom_surface_has_no_compatibility_aliases():
    from fatqat.emulator import atom_2level

    removed_names = (
        "Analog" + "Atom" + "Emulator",
        "Analog" + "Atom" + "Model",
    )
    for owner in (fq.emulator, atom_2level):
        for name in removed_names:
            assert not hasattr(owner, name)
            assert name not in owner.__all__
    assert importlib.util.find_spec("fatqat.emulator." + "atom") is None
    assert importlib.util.find_spec("fatqat.emulator.atom_2level.policy") is None


def test_atom_2level_constructor_has_final_public_keywords():
    parameters = inspect.signature(Atom2LevelEmulator).parameters
    assert tuple(parameters) == (
        "model",
        "arrangement",
        "method",
        "noise",
        "gate_implementation_map",
    )
    assert parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in (
            "arrangement",
            "method",
            "noise",
            "gate_implementation_map",
        )
    )
    assert parameters["arrangement"].default is inspect.Parameter.empty
    assert parameters["method"].default == "statevector"
    assert parameters["noise"].default is None
    assert parameters["gate_implementation_map"].default is None
    assert issubclass(Atom2LevelEmulator, _PulseBackend)


def test_public_family_constructors_reject_removed_lindblad_keyword(model):
    atom_2level_model = Atom2LevelModel.from_document(
        fq.emulator.load_model_document("atom2level.reference")
    )
    arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
    constructors = (
        lambda: TransmonEmulator(model, lindblad_implementation_map=object()),
        lambda: Atom2LevelEmulator(
            atom_2level_model,
            arrangement=arrangement,
            lindblad_implementation_map=object(),
        ),
    )

    for construct in constructors:
        with pytest.raises(TypeError, match="lindblad_implementation_map"):
            construct()


def test_atom_2level_execution_methods_have_exact_public_forwarding_signatures():
    run = inspect.signature(Atom2LevelEmulator.run).parameters
    assert tuple(run) == (
        "self",
        "program",
        "shots",
        "resource_layout",
        "simulation_config",
        "result_config",
    )
    assert run["program"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        run[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("shots", "resource_layout", "simulation_config", "result_config")
    )
    assert run["shots"].default == 1024
    assert run["resource_layout"].default is None
    assert run["simulation_config"].default is None
    assert run["result_config"].default is None
    assert get_type_hints(Atom2LevelEmulator.run)["return"] == Job[fq.Result]

    assert not hasattr(Atom2LevelEmulator, "propagator")
    assert not hasattr(Atom2LevelEmulator, "apply_final_frame")


def test_pulse_authoring_values_are_public_and_identical_to_private_definitions():
    assert fq.emulator.PulseDefinition is PulseDefinition is _PulseDefinition
    assert fq.emulator.PulseControl is PulseControl is _PulseControl
    assert fq.emulator.ControlChannel is ControlChannel is _ControlChannel
    removed_name = "Sampled" + "Control"
    assert not hasattr(fq.emulator, removed_name)
    assert removed_name not in fq.emulator.__all__
    assert fq.emulator.PhaseShift is PhaseShift is _PhaseShift
    assert fq.emulator.PhaseSwap is PhaseSwap is _PhaseSwap
    assert (
        fq.emulator.PulseImplementationMap
        is PulseImplementationMap
        is _PulseImplementationMap
    )
    assert (
        fq.emulator.default_transmon_gate_implementation_map
        is default_transmon_gate_implementation_map
    )
    assert fq.DeviceOperand is DeviceOperand

    definition = inspect.signature(PulseDefinition).parameters
    assert tuple(definition) == ("duration", "controls", "post_actions")
    assert definition["post_actions"].default == ()


def test_models_author_structural_controls_and_frames_without_public_handles(model):
    document_path = (
        Path(__file__).parents[1]
        / "atom_2level"
        / "fixtures"
        / "atom_2level_reference.json"
    )
    atom_2level_model = Atom2LevelModel.from_document(
        json.loads(document_path.read_text(encoding="utf-8"))
    )
    controls = (
        model.control.drive("q0"),
        model.control.exchange("q0", "q1"),
        atom_2level_model.control.drive(),
        atom_2level_model.control.detuning(),
    )
    assert all(isinstance(control, ControlChannel) for control in controls)
    frame = model.frame("q0")
    assert PhaseShift(frame, 0.1).frame is frame


def test_model_control_discovery_is_minimal_immutable_and_family_owned(model):
    atom_2level_model = Atom2LevelModel.from_document(
        fq.emulator.load_model_document("atom2level.reference")
    )
    families = (
        (
            atom_2level_model,
            {
                "drive": ("global", (), "complex", "rad/us"),
                "detuning": ("global", (), "real", "rad/us"),
            },
        ),
        (
            model,
            {
                "drive": (
                    "local",
                    ("subsystem_id",),
                    "complex",
                    "rad/ns",
                ),
                "detuning": (
                    "local",
                    ("subsystem_id",),
                    "real",
                    "rad/ns",
                ),
                "exchange": (
                    "pair",
                    ("first", "second"),
                    "real",
                    "rad/ns",
                ),
            },
        ),
    )

    for physics_model, expected in families:
        assert tuple(physics_model.available_controls) == tuple(expected)
        for name, metadata in expected.items():
            selector = physics_model.available_controls[name]
            assert selector is getattr(physics_model.control, name)
            assert (
                selector.scope,
                selector.operands,
                selector.coefficient_domain,
                selector.coefficient_unit,
            ) == metadata
        with pytest.raises(TypeError):
            physics_model.available_controls["other"] = object()


def test_concrete_target_owned_claim_classes_are_not_exported():
    from fatqat.emulator import superconducting

    for owner in (fq.emulator, superconducting):
        for name in (
            "SubsystemResourceRef",
            "CouplingRef",
            "ControlChannel" + "Ref",
            "Frame" + "Ref",
        ):
            assert not hasattr(owner, name)
            assert name not in owner.__all__
