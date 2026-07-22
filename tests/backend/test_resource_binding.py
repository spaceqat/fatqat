"""Tests the internal resource-binding infrastructure: `BoundResource`,
`ResourceBinding` dispatch, the scalar/identity binder, and the
`SimulatorBackend` hook that installs it.
"""

import pytest

from fatqat import operations as ops
from fatqat.backends import SimulatorBackend
from fatqat.backends.resource_binding import (
    BoundResource,
    ResourceBinding,
    _scalar_identity_binder,
)
from fatqat.errors import BackendValidationError, UnsupportedResourceOperandError
from fatqat.program import Program
from fatqat.registers import GridRegister


# --- BoundResource -----------------------------------------------------------


def test_bound_resource_is_immutable():
    p = Program(1)
    ref = p.qreg[0][0]
    bound = BoundResource(ref=ref, engine_index=0, device_label=0)
    with pytest.raises(AttributeError):
        bound.engine_index = 1


# --- scalar/identity binder ---------------------------------------------------


def test_scalar_identity_binder_maps_ref_to_matching_engine_index_and_device_label():
    p = Program(2)
    ref = p.qreg[0][1]
    layout = SimulatorBackend().resolve_layout(p)
    bound = _scalar_identity_binder(ref, layout)
    assert bound == BoundResource(ref=ref, engine_index=1, device_label=1)


def test_scalar_identity_binder_declines_register_view():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    layout = SimulatorBackend().resolve_layout(p)
    assert _scalar_identity_binder(atoms.row(0), layout) is None


# --- ResourceBinding dispatch --------------------------------------------------


def test_resource_binding_first_non_decline_wins():
    p = Program(1)
    ref = p.qreg[0][0]
    layout = SimulatorBackend().resolve_layout(p)
    sentinel = BoundResource(ref=ref, engine_index=99, device_label="x")
    binding = ResourceBinding([lambda t, l: sentinel, _scalar_identity_binder])
    assert binding.resolve(ref, layout) is sentinel


def test_resource_binding_tries_next_binder_when_first_declines():
    p = Program(1)
    ref = p.qreg[0][0]
    layout = SimulatorBackend().resolve_layout(p)
    binding = ResourceBinding([lambda t, l: None, _scalar_identity_binder])
    bound = binding.resolve(ref, layout)
    assert bound == BoundResource(ref=ref, engine_index=0, device_label=0)


def test_resource_binding_raises_unsupported_resource_operand_when_no_binder_resolves():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    layout = SimulatorBackend().resolve_layout(p)
    binding = ResourceBinding([_scalar_identity_binder])
    with pytest.raises(UnsupportedResourceOperandError):
        binding.resolve(atoms.row(0), layout)


def test_unsupported_resource_operand_error_is_backend_validation_error():
    assert issubclass(UnsupportedResourceOperandError, BackendValidationError)


# --- SimulatorBackend integration ----------------------------------------------


def test_simulator_backend_default_binding_is_scalar_identity():
    p = Program(2)
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    binding = backend._create_resource_binding(p, layout)
    ref = p.qreg[0][1]
    bound = binding.resolve(ref, layout)
    assert bound.engine_index == bound.device_label == 1


def test_simulator_backend_rejects_register_view():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.RX(0.3), atoms.row(0))
    with pytest.raises(UnsupportedResourceOperandError):
        SimulatorBackend().run(p)


def test_lower_accepts_explicit_binding_argument():
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    binding = backend._create_resource_binding(p, layout)
    plan, _facts = backend._lower(p, layout, binding)
    assert plan[0].target_indices == (0, 1)


def test_lower_without_binding_argument_builds_its_own():
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout)
    assert plan[0].target_indices == (0, 1)
