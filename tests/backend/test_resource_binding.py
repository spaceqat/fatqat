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
from fatqat.implementation import ImplementationMap, default_matrix_implementation_map
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


def test_lower_uses_device_label_for_lookup_and_engine_index_for_step():
    """Regression test for the device_labels/engine_indices swap risk flagged
    in review: every existing test binds identity (device_label ==
    engine_index numerically), so a swap at either `_lower` call site would
    go undetected. Here the two deliberately differ: the implementation map
    only has a rule keyed by the *device labels* (99, 100), not by the
    *engine indices* (0, 1), so `_lower` only succeeds at all if
    `_implementation_for` is called with device labels. The resulting
    `ApplyMatrixStep.target_indices` must then be the engine indices, not the
    device labels that were used for the lookup.
    """
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    layout = SimulatorBackend().resolve_layout(p)

    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    impl_map = ImplementationMap()
    impl_map.add(ops.CZ, cz_rule, device_operands=(99, 100))
    backend = SimulatorBackend(implementation_map=impl_map)

    q0, q1 = p.qreg[0][0], p.qreg[0][1]
    mismatched = {
        q0: BoundResource(ref=q0, engine_index=0, device_label=99),
        q1: BoundResource(ref=q1, engine_index=1, device_label=100),
    }
    binding = ResourceBinding([lambda t, l: mismatched[t]])

    plan, _facts = backend._lower(p, layout, binding)

    assert plan[0].target_indices == (0, 1)
