"""Tests the internal resource-binding infrastructure: `BoundResource`,
`ResourceBinding` dispatch, the scalar/identity binder, and the
`SimulatorBackend` hook that installs it.
"""

import pytest

from fatqat import operations as ops
from fatqat.backends import ApplyMatrixStep, SimulatorBackend
from fatqat.backends.resource_binding import (
    BoundResource,
    ResourceBinding,
    _scalar_identity_binder,
)
from fatqat.errors import BackendValidationError, UnsupportedResourceOperandError
from fatqat.implementation import ImplementationMap, default_matrix_implementation_map
from fatqat.operations import Measurement
from fatqat.program import AppliedOperation, Program
from fatqat.registers import (
    AllSelector,
    BlockSelector,
    ColumnSelector,
    GridRegister,
    RegisterView,
    RowSelector,
)


# --- test-only view binder ----------------------------------------------------
#
# Task 6 (FakeAtomGridBackend's grid binder) is what will resolve a real
# `RegisterView` to a tuple of `BoundResource`s in production. This task only
# owns the lowering-side expansion machinery, so the tests here drive it with a
# synthetic binder that walks the view's selector itself, in the deterministic
# member order `RegisterView` documents, and binds each member to identity
# engine-index/device-label values (like `_scalar_identity_binder` does for a
# scalar ref).


def _view_member_refs(view):
    """Enumerate a view's member refs in its documented deterministic order.

    Row-major for `All`/`Block`, increasing-column for `Row`, increasing-row
    for `Column` - matching `RegisterView`'s docstring. A `GridRegister`
    flattens row-major (flat index = row * cols + col).
    """
    reg = view.register
    sel = view.selector
    if isinstance(sel, AllSelector):
        return [reg[i] for i in range(reg.size)]
    if isinstance(sel, RowSelector):
        return [reg[sel.row * reg.cols + c] for c in range(reg.cols)]
    if isinstance(sel, ColumnSelector):
        return [reg[r * reg.cols + sel.col] for r in range(reg.rows)]
    if isinstance(sel, BlockSelector):
        (r0, r1), (c0, c1) = sel.rows, sel.cols
        return [
            reg[r * reg.cols + c] for r in range(r0, r1) for c in range(c0, c1)
        ]
    raise AssertionError(f"unhandled selector {sel!r}")


def _view_binder(target, flat_layout):
    """Resolve a `RegisterView` to a tuple of identity `BoundResource`s.

    Declines (returns `None`) for a scalar `RegisterRef` so it can be chained
    ahead of `_scalar_identity_binder`, exactly as a real grid binder would.
    """
    if not isinstance(target, RegisterView):
        return None
    return tuple(
        BoundResource(
            ref=m,
            engine_index=flat_layout.subsystem_index(m),
            device_label=flat_layout.subsystem_index(m),
        )
        for m in _view_member_refs(target)
    )


def _view_binding():
    return ResourceBinding([_view_binder, _scalar_identity_binder])


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


# --- view expansion during lowering -------------------------------------------


def _matrix_steps(plan):
    return [s for s in plan if isinstance(s, ApplyMatrixStep)]


def test_scalar_rotation_still_emits_exactly_one_step():
    # Regression: a genuinely scalar instruction must expand to exactly one
    # ApplyMatrixStep with the scalar's own engine index, byte-identical to
    # pre-Task-5 behavior.
    p = Program(2)
    p.add(ops.RX(0.3), 1)
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout, _view_binding())
    steps = _matrix_steps(plan)
    assert len(steps) == 1
    assert steps[0].target_indices == (1,)


def test_viewed_rotation_emits_one_step_per_member_in_order():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.RX(0.3), atoms.row(0))  # members: atoms[0], atoms[1], atoms[2]
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout, _view_binding())
    steps = _matrix_steps(plan)
    assert [s.target_indices for s in steps] == [(0,), (1,), (2,)]


def test_viewed_rotation_over_column_emits_increasing_row_order():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.RY(0.5), atoms.column(1))  # members: atoms[1], atoms[4]
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout, _view_binding())
    steps = _matrix_steps(plan)
    assert [s.target_indices for s in steps] == [(1,), (4,)]


def test_equal_cardinality_cx_views_zip_control_then_target_in_order():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    # control = row 0 (atoms[0], atoms[1]); target = row 1 (atoms[2], atoms[3])
    p.add(ops.CX, (atoms.row(0), atoms.row(1)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout, _view_binding())
    steps = _matrix_steps(plan)
    # Pair i zips control[i] with target[i]: (0,2) then (1,3), control first.
    assert [s.target_indices for s in steps] == [(0, 2), (1, 3)]


def test_equal_cardinality_cz_views_zip_in_order():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.CZ, (atoms.column(0), atoms.column(1)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout, _view_binding())
    steps = _matrix_steps(plan)
    # column(0) = atoms[0], atoms[2]; column(1) = atoms[1], atoms[3].
    assert [s.target_indices for s in steps] == [(0, 1), (2, 3)]


def test_unequal_cardinality_views_raise_before_any_step_appended():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    # row(0) has 3 members, column(0) has 2 members.
    p.add(ops.CX, (atoms.row(0), atoms.column(0)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._lower(p, layout, _view_binding())


def test_scalar_view_mixture_raises():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    # First operand is a view, second a scalar ref: never a valid pairing.
    p.add(ops.CX, (atoms.row(1), atoms[0]))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._lower(p, layout, _view_binding())


def test_view_scalar_mixture_raises_other_order():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.CX, (atoms[0], atoms.row(1)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._lower(p, layout, _view_binding())


def test_self_pair_within_zip_raises():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    # row(0) = atoms[0], atoms[1]; column(0) = atoms[0], atoms[2].
    # Zip position 0 pairs atoms[0] with atoms[0]: a self-pair.
    p.add(ops.CZ, (atoms.row(0), atoms.column(0)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._lower(p, layout, _view_binding())


def test_every_emitted_step_inherits_the_source_condition():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.RX(0.4), atoms.all(), condition=(0, 1))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    plan, _facts = backend._lower(p, layout, _view_binding())
    steps = _matrix_steps(plan)
    assert len(steps) == 4  # 2x2 = 4 members
    expected_cond = ((layout.clbit_index(p.clreg[0][0]), 1),)
    assert all(s.condition == expected_cond for s in steps)
    assert all(s.condition is not None for s in steps)


# --- _break_grouped_operations pre-lowering pass ------------------------------


def test_break_grouped_operations_passes_scalar_only_instructions_unchanged():
    # A scalar-only program: every instruction passes through identically
    # (same objects, 1:1, order preserved), no expansion.
    p = Program(2, 2)
    p.add(ops.RX(0.3), 1)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    binding = backend._create_resource_binding(p, layout)
    broken = backend._break_grouped_operations(p, layout, binding)
    assert broken == list(p.operations)
    assert all(a is b for a, b in zip(broken, p.operations))


def test_break_grouped_operations_passes_measurement_through():
    p = Program(1, 1)
    p.add_measurement(0, 0)
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    binding = backend._create_resource_binding(p, layout)
    broken = backend._break_grouped_operations(p, layout, binding)
    assert len(broken) == 1
    assert isinstance(broken[0], Measurement)
    assert broken[0] is p.operations[0]


def test_break_grouped_operations_expands_viewed_rotation_in_order():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.RX(0.3), atoms.row(0))  # members: atoms[0], atoms[1], atoms[2]
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    broken = backend._break_grouped_operations(p, layout, _view_binding())
    assert len(broken) == 3
    assert all(isinstance(op, AppliedOperation) for op in broken)
    assert [op.targets for op in broken] == [
        (atoms[0],),
        (atoms[1],),
        (atoms[2],),
    ]
    assert all(op.operation is p.operations[0].operation for op in broken)


def test_break_grouped_operations_zips_paired_cx_view_in_order():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    # control = row 0 (atoms[0], atoms[1]); target = row 1 (atoms[2], atoms[3])
    p.add(ops.CX, (atoms.row(0), atoms.row(1)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    broken = backend._break_grouped_operations(p, layout, _view_binding())
    assert [op.targets for op in broken] == [
        (atoms[0], atoms[2]),
        (atoms[1], atoms[3]),
    ]


def test_break_grouped_operations_preserves_condition():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.RX(0.4), atoms.all(), condition=(0, 1))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    broken = backend._break_grouped_operations(p, layout, _view_binding())
    assert len(broken) == 4
    source_cond = p.operations[0].condition
    assert all(op.condition == source_cond for op in broken)
    assert all(op.condition is not None for op in broken)


def test_break_grouped_operations_rejects_unequal_cardinality():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.CX, (atoms.row(0), atoms.column(0)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._break_grouped_operations(p, layout, _view_binding())


def test_break_grouped_operations_rejects_scalar_view_mixture():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.CX, (atoms.row(1), atoms[0]))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._break_grouped_operations(p, layout, _view_binding())


def test_break_grouped_operations_rejects_self_pair():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.CZ, (atoms.row(0), atoms.column(0)))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    with pytest.raises(BackendValidationError):
        backend._break_grouped_operations(p, layout, _view_binding())


def test_break_grouped_operations_does_not_mutate_program():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.RX(0.3), atoms.row(0))
    before = p.operations
    before_list = list(before)
    backend = SimulatorBackend()
    layout = backend.resolve_layout(p)
    backend._break_grouped_operations(p, layout, _view_binding())
    # The user-facing operations tuple is unchanged in identity and value.
    assert p.operations is before
    assert list(p.operations) == before_list
