"""NoiseModel admission, matching, conflict, and lifecycle contracts."""

import inspect

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    Loss,
    NoiseModel,
    PhaseDamping,
    ReadoutConfusion,
)
from fatqat.registers import GridRegister
from fatqat.resource_layout import ResourceLayout

_CONFUSION = ReadoutConfusion([[0.9, 0.2], [0.1, 0.8]])


def _program_and_layout(size=3):
    program = fq.Program(size)
    q = program.quantum_registers[0]
    return program, q, ResourceLayout({q[index]: index for index in range(size)})


def test_add_is_the_only_public_registration_verb_and_returns_none():
    noise = NoiseModel()

    assert noise.add(Depolarizing(p=0.1), operation=fq.ops.X) is None
    assert noise.add(_CONFUSION, targets=0) is None
    assert "declaration" in inspect.signature(noise.add).parameters
    for obsolete in (
        "add_channel",
        "add_readout_error",
        "channels_for",
        "always_on_channels_for",
        "readout_error_for",
        "channel_registrations",
        "metadata",
        "_channels",
        "_readout_errors",
    ):
        assert not hasattr(noise, obsolete)


@pytest.mark.parametrize("keyword", ["operation", "target_positions"])
@pytest.mark.parametrize("value", [None, fq.ops.X])
def test_readout_rejects_dynamical_scope_keywords_atomically(keyword, value):
    noise = NoiseModel()

    with pytest.raises(TypeError):
        noise.add(_CONFUSION, **{keyword: value})

    assert noise._readout_confusions() == ()


def test_readout_rejects_tuple_targets_even_when_length_one():
    noise = NoiseModel()

    with pytest.raises(TypeError, match="scalar"):
        noise.add(_CONFUSION, targets=(0,))


def test_operation_is_occurrence_scope_and_omission_is_background_scope():
    noise = NoiseModel()
    operation_channel = PhaseDamping(p=0.1)
    background_channel = PhaseDamping(rate=2.0)
    noise.add(operation_channel, operation=fq.ops.X)
    noise.add(background_channel, targets=1)
    _program, q, layout = _program_and_layout()

    assert noise._noise_for_occurrence(fq.ops.X, (q[0],), layout) == [
        (operation_channel, (q[0],))
    ]
    assert noise._background_noise_for(q[1], 1) == (background_channel,)
    assert noise._background_noise_for(q[0], 0) == ()


def test_operationless_noise_is_not_an_every_gate_shorthand():
    with pytest.raises(ValueError, match="not shorthand"):
        NoiseModel().add(PhaseDamping(p=0.1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation": fq.ops.X, "targets": (None,)},
        {"targets": (None,)},
    ],
)
def test_none_cannot_be_a_physical_dynamical_selector(kwargs):
    with pytest.raises(TypeError, match="cannot be None"):
        NoiseModel().add(PhaseDamping(rate=1.0), **kwargs)


@pytest.mark.parametrize(
    "declaration, kwargs",
    [
        (Loss(p=0.1), {"targets": 0}),
        (Depolarizing(p=0.1), {"targets": 0}),
        (PhaseDamping(rate=1.0), {"targets": (0, 1)}),
        (PhaseDamping(rate=1.0), {"targets": 0, "target_positions": 0}),
    ],
)
def test_background_scope_rejects_nonlocal_or_positional_forms(declaration, kwargs):
    with pytest.raises(ValueError):
        NoiseModel().add(declaration, **kwargs)


@pytest.mark.parametrize(
    "operation", [fq.ops.Barrier, fq.ops.LoadAtoms(1, 1), fq.ops.Reset]
)
def test_operations_without_noise_boundaries_reject_atomically(operation):
    noise = NoiseModel()

    with pytest.raises(ValueError):
        noise.add(Depolarizing(p=0.1), operation=operation)

    assert noise._noise_sources() == ()


def test_occurrence_selector_is_exact_ordered_and_positions_select_extent():
    _program, q, layout = _program_and_layout()
    damping = AmplitudeDamping(p=0.1)
    noise = NoiseModel()
    noise.add(
        damping,
        operation=fq.ops.CX,
        targets=(q[0], q[1]),
        target_positions=1,
    )

    assert noise._noise_for_occurrence(fq.ops.CX, (q[0], q[1]), layout) == [
        (damping, (q[1],))
    ]
    assert noise._noise_for_occurrence(fq.ops.CX, (q[1], q[0]), layout) == []


def test_physical_selector_matches_exact_order_and_tuple_label_is_opaque():
    _program, q, _layout = _program_and_layout(2)
    layout = ResourceLayout({q[0]: ("site", 0), q[1]: ("site", 1)})
    channel = Depolarizing(p=0.1)
    noise = NoiseModel()
    noise.add(
        channel,
        operation=fq.ops.CX,
        targets=(("site", 0), ("site", 1)),
    )

    assert noise._noise_for_occurrence(fq.ops.CX, (q[0], q[1]), layout) == [
        (channel, (q[0], q[1]))
    ]


def test_scalar_selector_is_unary_shorthand_only():
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.X, targets="q0")
    with pytest.raises(ValueError, match="length"):
        noise.add(Depolarizing(p=0.2), operation=fq.ops.CX, targets="q0")


@pytest.mark.parametrize(
    "second, conflicts",
    [
        ({}, True),
        ({"targets": (0, 1)}, True),
        ({"target_positions": 0}, True),
        ({"target_positions": 1}, False),
        ({"targets": (0, 1), "target_positions": 0}, True),
    ],
)
def test_same_source_overlap_matrix(second, conflicts):
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.CX, target_positions=0)

    def call():
        noise.add(Depolarizing(p=0.2), operation=fq.ops.CX, **second)

    if conflicts:
        with pytest.raises(ValueError, match="overlapping"):
            call()
    else:
        call()


def test_different_sources_and_activation_scopes_accumulate():
    noise = NoiseModel()
    noise.add(PhaseDamping(p=0.1), operation=fq.ops.X)
    noise.add(AmplitudeDamping(p=0.1), operation=fq.ops.X)
    noise.add(PhaseDamping(rate=1.0), targets=0)

    assert len(noise._noise_sources()) == 3


def test_distinct_exact_selectors_support_heterogeneous_calibration():
    noise = NoiseModel()
    noise.add(PhaseDamping(p=0.1), operation=fq.ops.X, targets=0)
    noise.add(PhaseDamping(p=0.2), operation=fq.ops.X, targets=1)

    assert len(noise._noise_sources()) == 2


def test_logical_and_physical_alias_conflicts_only_on_actual_match():
    _program, q, layout = _program_and_layout()
    noise = NoiseModel()
    noise.add(PhaseDamping(p=0.1), operation=fq.ops.X, targets=q[0])
    noise.add(PhaseDamping(p=0.2), operation=fq.ops.X, targets=0)

    assert noise._noise_for_occurrence(fq.ops.X, (q[1],), layout) == []
    with pytest.raises(BackendValidationError, match="both match"):
        noise._noise_for_occurrence(fq.ops.X, (q[0],), layout)


def test_logical_physical_alias_is_valid_across_disjoint_positions():
    _program, q, layout = _program_and_layout()
    noise = NoiseModel()
    noise.add(
        PhaseDamping(p=0.1),
        operation=fq.ops.CX,
        targets=(q[0], q[1]),
        target_positions=0,
    )
    noise.add(
        PhaseDamping(p=0.2),
        operation=fq.ops.CX,
        targets=(0, 1),
        target_positions=1,
    )

    assert len(noise._noise_for_occurrence(fq.ops.CX, (q[0], q[1]), layout)) == 2


def test_readout_is_unique_per_operand_without_replacement():
    noise = NoiseModel()
    noise.add(_CONFUSION, targets=0)

    with pytest.raises(ValueError, match="already registered"):
        noise.add(ReadoutConfusion(np.eye(2)), targets=0)


def test_equal_physical_readout_labels_reject_regardless_of_python_type():
    noise = NoiseModel()
    noise.add(_CONFUSION, targets=1)

    with pytest.raises(ValueError, match="already registered"):
        noise.add(ReadoutConfusion(np.eye(2)), targets=True)


def test_universal_and_targeted_readout_cannot_coexist():
    noise = NoiseModel()
    noise.add(_CONFUSION)

    with pytest.raises(ValueError, match="cannot coexist"):
        noise.add(ReadoutConfusion(np.eye(2)), targets=0)


def test_readout_logical_physical_alias_rejects_at_measurement():
    _program, q, layout = _program_and_layout()
    noise = NoiseModel()
    noise.add(_CONFUSION, targets=q[0])
    noise.add(ReadoutConfusion(np.eye(2)), targets=0)

    assert noise._readout_confusion_for(q[1], layout) is None
    with pytest.raises(BackendValidationError, match="multiple"):
        noise._readout_confusion_for(q[0], layout)


def test_validate_for_checks_logical_ownership_and_physical_universe():
    program, q, layout = _program_and_layout()
    foreign = fq.QuantumRegister(1)[0]
    logical = NoiseModel()
    logical.add(PhaseDamping(p=0.1), operation=fq.ops.X, targets=foreign)
    with pytest.raises(BackendValidationError, match="outside this program"):
        logical._validate_for(program, frozenset(layout.device_labels))

    physical = NoiseModel()
    physical.add(PhaseDamping(p=0.1), operation=fq.ops.X, targets=99)
    with pytest.raises(BackendValidationError, match="legal universe"):
        physical._validate_for(program, frozenset(layout.device_labels))

    valid_no_match = NoiseModel()
    valid_no_match.add(PhaseDamping(p=0.1), operation=fq.ops.Y, targets=q[2])
    valid_no_match._validate_for(program, frozenset(layout.device_labels))


def test_copy_owns_independent_lists_but_shares_immutable_values():
    channel = PhaseDamping(p=0.1)
    confusion = ReadoutConfusion(np.eye(2))
    source = NoiseModel()
    source.add(channel, operation=fq.ops.X)
    source.add(confusion, targets=0)
    copied = source._copy()
    source.add(AmplitudeDamping(p=0.2), operation=fq.ops.Y)

    assert copied._noise_sources() == ((channel, type(fq.ops.X)),)
    assert copied._readout_confusions()[0] is confusion
    assert len(source._noise_sources()) == 2


def test_register_view_and_mixed_identity_selectors_reject():
    atoms = GridRegister(2, 2)
    noise = NoiseModel()
    with pytest.raises(TypeError, match="RegisterView"):
        noise.add(Depolarizing(p=0.1), operation=fq.ops.CX, targets=atoms.row(0))
    with pytest.raises(TypeError, match="all RegisterRef"):
        noise.add(Depolarizing(p=0.1), operation=fq.ops.CX, targets=(atoms[0], 1))
