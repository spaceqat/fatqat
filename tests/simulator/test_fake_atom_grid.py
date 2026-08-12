"""Tests the fake atom grid backend's grid binding, device-shape, and native
gate-set constraints.
"""

import numpy as np
import pytest

from fatqat import operations as ops
from fatqat._backends.steps import ApplyMatrixStep, ResetStep, RefillStep
from fatqat.simulator import AtomGridSimulator, Simulator
from fatqat.simulator.fake_atom_grid import fake_atom_grid_implementation_map
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import Depolarizing, NoiseModel, AtomLoss
from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister
from fatqat.resource_layout import ResourceLayout


def _matrix_steps(plan):
    return [step for step in plan if isinstance(step, ApplyMatrixStep)]


# --- constructor validation / default shape -----------------------------------


def test_default_shape_is_4x5():
    backend = AtomGridSimulator()
    p_fits = Program(20)
    p_fits.add(ops.RX(0.1), 0)
    # 4x5 = 20 qubits fits exactly.
    layout = backend._allocate_engine_indices(p_fits)
    assert layout.n_subsystems == 20

    resource_layout = backend._resolve_resource_layout(p_fits)
    assert isinstance(resource_layout, ResourceLayout)

    p_too_big = Program(21)
    p_too_big.add(ops.RX(0.1), 0)
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p_too_big)


def test_rejects_non_tuple_grid_size():
    with pytest.raises(TypeError):
        AtomGridSimulator(grid_size=[2, 3])


def test_rejects_grid_size_with_wrong_length():
    with pytest.raises(ValueError):
        AtomGridSimulator(grid_size=(2,))


def test_rejects_non_int_grid_entry():
    with pytest.raises(TypeError):
        AtomGridSimulator(grid_size=(2, "3"))


def test_rejects_bool_grid_row():
    with pytest.raises(TypeError):
        AtomGridSimulator(grid_size=(True, 3))


def test_rejects_bool_grid_column():
    with pytest.raises(TypeError):
        AtomGridSimulator(grid_size=(2, False))


def test_rejects_zero_grid_row():
    with pytest.raises(ValueError):
        AtomGridSimulator(grid_size=(0, 3))


def test_rejects_negative_grid_column():
    with pytest.raises(ValueError):
        AtomGridSimulator(grid_size=(2, -1))


# --- 2x3-on-4x5 acceptance example ---------------------------------------------


def test_2x3_grid_binds_top_left_on_default_4x5_backend():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.RX(0.1), atoms.all())

    backend = AtomGridSimulator()  # default 4x5
    engine_index_allocation = backend._allocate_engine_indices(p)
    resource_layout = backend._resolve_resource_layout(p)

    assert tuple(resource_layout.device_label(atoms[i]) for i in range(6)) == (
        0,
        1,
        2,
        5,
        6,
        7,
    )
    assert tuple(
        engine_index_allocation.subsystem_index(atoms[i]) for i in range(6)
    ) == tuple(range(6))


def test_scalar_grid_ref_uses_grid_binder_device_label_not_identity():
    # atoms[3] is frontend (row=1, col=0) on a 2x3 grid: engine index 3 (flat
    # layout order), but device label must come from the *backend*'s column
    # count (5), i.e. 1*5+0 = 5, not the identity value 3.
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    backend = AtomGridSimulator()  # default 4x5
    ref = atoms[3]
    engine_index_allocation = backend._allocate_engine_indices(p)
    resource_layout = backend._resolve_resource_layout(p)
    assert engine_index_allocation.subsystem_index(ref) == 3
    assert resource_layout.device_label(ref) == 5


# --- gate-channel noise selectors: physical labels, not engine indices --------


def test_physical_noise_selector_uses_device_label_not_engine_index():
    # 2x3 grid on the default 4x5 backend: atoms[3] is engine index 3, but
    # its device label is 5 (row 1, col 0 -> 1*5+0). A physical gate-noise
    # selector must be written in device-label space: (5,) selects atoms[3],
    # while (3,) - its old, no-longer-meaningful engine index - selects
    # nothing, since no program resource occupies device site 3.
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms])
    backend = AtomGridSimulator()
    resource_layout = backend._resolve_resource_layout(program)
    ref = atoms[3]
    assert resource_layout.device_label(ref) == 5

    channel = Depolarizing(p=0.1)
    noise = NoiseModel()
    noise.add_channel(channel, operation=ops.RX, targets=(5,))

    assert noise.channels_for(ops.RX, (ref,), resource_layout) == [(channel, (ref,))]

    stale_engine_index_selector = NoiseModel()
    stale_engine_index_selector.add_channel(channel, operation=ops.RX, targets=(3,))
    assert (
        stale_engine_index_selector.channels_for(ops.RX, (ref,), resource_layout) == []
    )


# --- readout-error selectors: physical labels, not engine indices --------------


def test_physical_readout_selector_uses_device_label_not_engine_index():
    # Same divergence as the gate-channel case: atoms[3] is engine index 3
    # but device label 5 on the default 4x5 backend. A physical readout
    # selector must be written in device-label space.
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms])
    backend = AtomGridSimulator()
    resource_layout = backend._resolve_resource_layout(program)
    ref = atoms[3]
    assert resource_layout.device_label(ref) == 5

    matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
    noise = NoiseModel()
    noise.add_readout_error(matrix, target=5)

    assert np.array_equal(noise.readout_error_for(ref, resource_layout), matrix)

    stale_engine_index_selector = NoiseModel()
    stale_engine_index_selector.add_readout_error(matrix, target=3)
    assert stale_engine_index_selector.readout_error_for(ref, resource_layout) is None


# --- sole-register / fit / multiplicity rules ----------------------------------
#
# All of these are resource-layout-level mapping/capacity concerns per the
# design, so they must raise from `_resolve_resource_layout()`, not from the
# generic engine-flattening `_allocate_engine_indices()`.


def test_rejects_grid_register_combined_with_other_quantum_register():
    atoms = GridRegister(2, 2, name="atoms")
    other = QuantumRegister(2, name="q")
    p = Program([atoms, other])
    backend = AtomGridSimulator()
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p)


def test_rejects_two_grid_registers():
    atoms1 = GridRegister(2, 2, name="a1")
    atoms2 = GridRegister(2, 2, name="a2")
    p = Program([atoms1, atoms2])
    backend = AtomGridSimulator()
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p)


def test_rejects_grid_that_does_not_fit_backend_even_with_enough_total_capacity():
    # 5x4 = 20 members, same total capacity as the default 4x5 backend, but
    # rows=5 exceeds the backend's 4 rows: must be rejected on a per-axis
    # basis, not just total member count.
    atoms = GridRegister(5, 4, name="atoms")
    p = Program([atoms])
    backend = AtomGridSimulator()
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p)


def test_resolve_resource_layout_rejects_capacity_for_scalar_only_program():
    p = Program(21)
    backend = AtomGridSimulator()  # capacity 20
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p)


def test_resolve_resource_layout_rejects_non_qubit_dimension_for_scalar_only_program():
    p = Program([QuantumRegister(4, dim=3)])
    backend = AtomGridSimulator()
    with pytest.raises(BackendValidationError, match="qubit dimensions"):
        backend._resolve_resource_layout(p)


def test_scalar_only_program_uses_identity_binding():
    p = Program(3)
    backend = AtomGridSimulator()
    ref = p.quantum_registers[0][2]
    engine_index_allocation = backend._allocate_engine_indices(p)
    resource_layout = backend._resolve_resource_layout(p)
    assert engine_index_allocation.subsystem_index(ref) == 2
    assert resource_layout.device_label(ref) == 2


def test_allocate_engine_has_no_atom_grid_validation_left():
    # `_allocate_engine_indices()` now just delegates to the generic engine-
    # flattening behavior: an over-capacity, non-qubit-dim, or ill-shaped
    # program flattens without complaint here (validation moved wholesale to
    # `_resolve_resource_layout()`).
    backend = AtomGridSimulator()  # capacity 20

    p_too_big = Program(21)
    engine_index_allocation = backend._allocate_engine_indices(p_too_big)
    assert engine_index_allocation.n_subsystems == 21

    p_bad_dim = Program([QuantumRegister(4, dim=3)])
    engine_index_allocation = backend._allocate_engine_indices(p_bad_dim)
    assert engine_index_allocation.n_subsystems == 4

    atoms1 = GridRegister(2, 2, name="a1")
    atoms2 = GridRegister(2, 2, name="a2")
    p_two_grids = Program([atoms1, atoms2])
    engine_index_allocation = backend._allocate_engine_indices(p_two_grids)
    assert engine_index_allocation.n_subsystems == 8


# --- implementation_map capability API -----------------------------------------


def test_implementation_map_exposes_four_native_families():
    m = AtomGridSimulator().implementation_map
    assert m.supports(ops.RX) and not m.device_operands_for(ops.RX)
    assert m.supports(ops.RY) and not m.device_operands_for(ops.RY)
    assert m.supports(ops.RZ) and not m.device_operands_for(ops.RZ)
    assert m.supports(ops.CZ)
    assert not m.supports(ops.CX)

    # Map introspection alone doesn't prove .run() actually rejects it -
    # exercise the real end-to-end path too.
    p = Program(2)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.CX, (0, 1))
    with pytest.raises(UnsupportedOperationError):
        AtomGridSimulator(grid_size=(1, 2)).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_implementation_map_cz_accepts_neighbor_pairs_both_directions():
    m = AtomGridSimulator().implementation_map  # default 4x5, backend cols=5
    assert (0, 1) in m.device_operands_for(ops.CZ)
    assert (1, 0) in m.device_operands_for(ops.CZ)
    assert (0, 5) in m.device_operands_for(ops.CZ)
    assert (5, 0) in m.device_operands_for(ops.CZ)


def test_implementation_map_rejects_non_neighbor_pairs():
    m = AtomGridSimulator().implementation_map
    assert (0, 2) not in m.device_operands_for(ops.CZ)
    assert (0, 6) not in m.device_operands_for(ops.CZ)


def test_fake_atom_grid_implementation_map_cz_has_no_class_keyed_rule():
    m = fake_atom_grid_implementation_map(4, 5)
    assert m.implementation_for(ops.CZ) is None


# --- resource-map behavior -----------------------------------------------------


def test_resource_layout_covers_all_scalar_refs_from_the_program_registers():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    backend = AtomGridSimulator()
    resource_layout = backend._resolve_resource_layout(p)
    assert {resource_layout.device_label(atoms[index]) for index in range(4)} == {
        0,
        1,
        5,
        6,
    }


# --- integration: numeric equivalence to manual scalar circuits ----------------


def test_viewed_rotation_matches_manual_scalar_sequence():
    atoms = GridRegister(1, 3, name="atoms")
    grid_p = Program([atoms])
    grid_p.add(ops.LoadAtoms(1, 3))
    grid_p.add(ops.RY(0.7), atoms.row(0))
    grid_sv = (
        AtomGridSimulator()
        .run(grid_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(3)
    for i in range(3):
        manual_p.add(ops.RY(0.7), i)
    manual_sv = (
        Simulator()
        .run(manual_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_paired_cz_view_matches_manual_scalar_sequence():
    # 2x2 grid on the default 4x5 backend: engine indices 0,1,2,3; device
    # labels 0,1,5,6 (backend cols=5). row(0) vs row(1) zips (atoms0,atoms2)
    # then (atoms1,atoms3) -> device pairs (0,5) and (1,6), both legal
    # vertical neighbor edges - even though the engine-index pairs (0,2) and
    # (1,3) are NOT adjacent under an identity mapping. This is the crux of
    # the task: device labels, not engine indices, drive gate legality.
    atoms = GridRegister(2, 2, name="atoms")
    grid_p = Program([atoms])
    grid_p.add(ops.LoadAtoms(2, 2))
    grid_p.add(ops.CZ, (atoms.row(0), atoms.row(1)))
    grid_sv = (
        AtomGridSimulator()
        .run(grid_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(4)
    manual_p.add(ops.CZ, (0, 2))
    manual_p.add(ops.CZ, (1, 3))
    manual_sv = (
        Simulator()
        .run(manual_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_viewed_rotation_over_column_matches_manual_scalar_sequence():
    # 2x3 grid: column(1) selects (row=0,col=1) and (row=1,col=1), in
    # increasing-row order -> flat/engine indices 1 and 4 (row-major,
    # index = row*cols+col). Exercises ColumnSelector through a real run(),
    # which only .all()/.row() had coverage for before this test.
    atoms = GridRegister(2, 3, name="atoms")
    grid_p = Program([atoms])
    grid_p.add(ops.LoadAtoms(2, 3))
    grid_p.add(ops.RY(0.7), atoms.column(1))
    grid_sv = (
        AtomGridSimulator()
        .run(grid_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(6)
    for i in (1, 4):
        manual_p.add(ops.RY(0.7), i)
    manual_sv = (
        Simulator()
        .run(manual_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_viewed_rotation_over_block_matches_manual_scalar_sequence():
    # 2x3 grid: block(rows=(0,2), cols=(1,3)) selects (0,1),(0,2),(1,1),(1,2)
    # in row-major order -> flat/engine indices 1, 2, 4, 5. Exercises
    # BlockSelector through a real run(), closing the same coverage gap as
    # the column test above.
    atoms = GridRegister(2, 3, name="atoms")
    grid_p = Program([atoms])
    grid_p.add(ops.LoadAtoms(2, 3))
    grid_p.add(ops.RY(0.7), atoms.block(rows=(0, 2), cols=(1, 3)))
    grid_sv = (
        AtomGridSimulator()
        .run(grid_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(6)
    for i in (1, 2, 4, 5):
        manual_p.add(ops.RY(0.7), i)
    manual_sv = (
        Simulator()
        .run(manual_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_non_neighbor_pair_rejects():
    atoms = GridRegister(2, 3, name="atoms")  # device labels row0:0,1,2 row1:5,6,7
    p = Program([atoms])
    p.add(ops.LoadAtoms(2, 3))
    p.add(ops.CZ, (atoms[0], atoms[5]))  # labels 0 and 7: not adjacent

    with pytest.raises(UnsupportedOperationError) as excinfo:
        AtomGridSimulator().run(p, result_config={"counts": False, "final_state": True})
    assert isinstance(excinfo.value, BackendValidationError)


def test_native_connectivity_lookup_uses_device_labels_not_engine_indices():
    # 2x3 grid on the default 4x5 backend: atoms[0] is engine index 0, device
    # label 0; atoms[3] is engine index 3, device label 5 (row 1, col 0 ->
    # 1*5+0). Labels (0, 5) are legal vertical neighbors on the 4x5 device,
    # but engine indices (0, 3) are NOT a legal edge on the same map (only
    # (0, 1)/(1, 2)-style row-adjacency or (i, i+5)-style column-adjacency
    # are). A CZ between atoms[0] and atoms[3] only succeeds if the
    # implementation-map lookup used the device labels (0, 5), not the
    # engine target indices (0, 3) it feeds into the execution step.
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.LoadAtoms(2, 3))
    p.add(ops.CZ, (atoms[0], atoms[3]))

    backend = AtomGridSimulator()
    assert (0, 5) in backend.implementation_map.device_operands_for(ops.CZ)
    assert (0, 3) not in backend.implementation_map.device_operands_for(ops.CZ)

    grid_sv = (
        backend.run(p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(6)
    manual_p.add(ops.CZ, (0, 3))
    manual_sv = (
        Simulator()
        .run(manual_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_lowering_uses_resource_layout_device_operands_and_engine_index_allocation():
    # 2x3 grid on the default 4x5 device: atoms[0] is engine index 0, device
    # label 0; atoms[3] is engine index 3, device label 5 (row 1, col 0 ->
    # 1*5+0). The native CZ map only legalizes the *device*-label edge
    # (0, 5), not the engine-index pair (0, 3), so lowering only succeeds by
    # looking up `MatrixImplementationMap` with device operands sourced from
    # `ResourceLayout`. The resulting `ApplyMatrixStep`, however, must carry
    # the *engine* indices (0, 3) from `_EngineIndexAllocation` - the private
    # lowering context keeps the two identities separate end to end.
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms])
    program.add(ops.LoadAtoms(2, 3))
    program.add(ops.CZ, (atoms[0], atoms[3]))

    backend = AtomGridSimulator()
    assert (0, 5) in backend.implementation_map.device_operands_for(ops.CZ)
    assert (0, 3) not in backend.implementation_map.device_operands_for(ops.CZ)

    plan, _facts = backend._lower_program(program)
    steps = _matrix_steps(plan)
    assert len(steps) == 1
    assert steps[0].target_indices == (0, 3)


def test_run_resolves_resource_layout_exactly_once_even_with_grid_mapping():
    # AtomGridSimulator's `_resolve_resource_layout` is non-trivial (grid
    # validation + top-left mapping); lowering must reuse the single value
    # `run()` already resolved rather than resolving it again for lookup.
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms])
    program.add(ops.LoadAtoms(2, 3))
    program.add(ops.RX(0.1), atoms[0])

    backend = AtomGridSimulator()
    calls = {"resource_layout": 0}
    original = backend._resolve_resource_layout

    def counting(program):
        calls["resource_layout"] += 1
        return original(program)

    backend._resolve_resource_layout = counting
    backend.run(program, result_config={"counts": False, "final_state": True})
    assert calls["resource_layout"] == 1


def test_device_label_for_method_is_gone():
    # The top-left mapping formula lives solely in `_resolve_resource_layout`
    # now; the old per-ref callback hook is removed from this backend.
    assert "_device_label_for" not in AtomGridSimulator.__dict__


def test_condition_on_viewed_instruction_propagates_end_to_end():
    atoms = GridRegister(1, 2, name="atoms")

    # Condition true (default clbit value is 0, condition checks == 0):
    # matches an unconditioned manual RX(pi) on both scalar qubits.
    grid_true = Program([atoms], 2)
    grid_true.add(ops.LoadAtoms(1, 2))
    grid_true.add(
        ops.RX(np.pi), atoms.all(), condition=(grid_true.classical_registers[0][0], 0)
    )
    sv_true = (
        AtomGridSimulator()
        .run(grid_true, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_true = Program(2)
    manual_true.add(ops.RX(np.pi), 0)
    manual_true.add(ops.RX(np.pi), 1)
    sv_manual_true = (
        Simulator()
        .run(manual_true, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(sv_true, sv_manual_true)

    # Condition false (clbit compared to 1, but default is 0): must not fire,
    # state stays |00>.
    grid_false = Program([atoms], 2)
    grid_false.add(ops.LoadAtoms(1, 2))
    grid_false.add(
        ops.RX(np.pi), atoms.all(), condition=(grid_false.classical_registers[0][0], 1)
    )
    sv_false = (
        AtomGridSimulator()
        .run(grid_false, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    expected_false = np.zeros(4, dtype=complex)
    expected_false[0] = 1.0
    assert np.allclose(sv_false, expected_false)

    # And the two runs must actually differ - proof the condition mattered.
    assert not np.allclose(sv_true, sv_false)


def test_grid_register_export_is_backend_public():
    # AtomGridSimulator is publicly exported; implementation details are not.
    import fatqat.simulator as simulator_pkg

    assert "AtomGridSimulator" in simulator_pkg.__all__
    assert not hasattr(simulator_pkg, "_build_qubit_resource_map")


# --- LoadAtoms lifecycle ---------------------------------------------------


def test_first_instruction_must_be_load_atom():
    p = Program(2)
    p.add(ops.X, 0)
    with pytest.raises(BackendValidationError, match="first"):
        AtomGridSimulator(grid_size=(1, 2)).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_empty_program_does_not_require_load_atom():
    # Falls out of the loop naturally: zero instructions means the "first
    # instruction must be LoadAtoms" check never runs, no special-cased branch
    # needed for it.
    p = Program(0, 0)
    plan, facts = AtomGridSimulator(grid_size=(1, 1))._lower_program(p)
    assert plan == []
    assert not facts.has_measurement


def test_second_load_atom_rejected():
    p = Program(2)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.LoadAtoms(1, 1))
    with pytest.raises(BackendValidationError, match="only as the"):
        AtomGridSimulator(grid_size=(1, 2)).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_load_atom_larger_than_device_rejected():
    p = Program(2)
    p.add(ops.LoadAtoms(2, 2))
    with pytest.raises(BackendValidationError, match="does not fit"):
        AtomGridSimulator(grid_size=(1, 2)).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_conditional_load_atom_rejected():
    p = Program(2, 1)
    p.add(ops.LoadAtoms(1, 2), condition=(p.classical_registers[0][0], 0))
    with pytest.raises(BackendValidationError, match="unconditional"):
        AtomGridSimulator(grid_size=(1, 2)).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_gate_on_unloaded_site_is_dropped():
    # AtomGridSimulator's native gate set is RX/RY/RZ/CZ only - X is not
    # native here (unlike the generic Simulator), so this uses RX(pi),
    # which matches X up to a global phase (see fq.ops.RX's own docstring
    # example), to stay within the native set.
    p = Program(2)
    p.add(ops.LoadAtoms(1, 1))  # only qubit 0's site is loaded
    p.add(ops.RX(np.pi), 0)
    p.add(ops.RX(np.pi), 1)  # site 1 unloaded: silently dropped
    sv = (
        AtomGridSimulator(grid_size=(1, 2))
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    expected = np.zeros(4, dtype=complex)
    expected[1] = -1j  # qubit0: RX(pi)|0> = -i|1>; qubit1 stays |0> (dropped)
    assert np.allclose(sv, expected)


def test_reset_on_unloaded_site_is_dropped_from_plan():
    # A statevector-only check can't distinguish "Reset ran on a qubit
    # that's already |0>" from "Reset was actually filtered": both leave the
    # same state. Assert directly against the lowered plan instead - no
    # ResetStep is produced at all, and facts.has_reset is False.
    p = Program(2)
    p.add(ops.LoadAtoms(1, 1))  # only qubit 0's site is loaded
    p.add(ops.Reset, 1)  # site 1 unloaded: must be dropped, never lowered
    plan, facts = AtomGridSimulator(grid_size=(1, 2))._lower_program(p)
    assert not facts.has_reset
    assert not any(isinstance(step, ResetStep) for step in plan)


def test_gate_on_loaded_site_executes_normally():
    p = Program(2)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.RX(0.3), 0)
    p.add(ops.RX(0.3), 1)
    grid_sv = (
        AtomGridSimulator(grid_size=(1, 2))
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual = Program(2)
    manual.add(ops.RX(0.3), 0)
    manual.add(ops.RX(0.3), 1)
    manual_sv = (
        Simulator()
        .run(manual, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(grid_sv, manual_sv)


def test_measurement_of_unloaded_site_reads_zero_without_noise():
    p = Program(2, 1)
    p.add(ops.LoadAtoms(1, 1))
    p.add(ops.X, 1)  # dropped: unloaded, so the site never actually flips
    p.measure(1, 0)
    counts = (
        AtomGridSimulator(grid_size=(1, 2))
        .run(p, shots=4, result_config={"counts": True}, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"0": 4}


def test_measurement_of_unloaded_site_still_exposed_to_readout_noise():
    # Deterministic "always report 1 for true 0" confusion matrix on device
    # label 1 (qubit index 1's site, left unloaded): pins that an unloaded
    # site's classical readout is NOT specially exempted from a configured
    # NoiseModel, even though its underlying quantum state is a clean |0>.
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = NoiseModel()
    noise.add_readout_error(matrix, target=1)

    p = Program(2, 1)
    p.add(ops.LoadAtoms(1, 1))
    p.measure(1, 0)
    counts = (
        AtomGridSimulator(grid_size=(1, 2), noise=noise)
        .run(p, shots=4, result_config={"counts": True}, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"1": 4}


def test_cz_becomes_legal_after_rearrange_makes_pair_adjacent():
    atoms = GridRegister(1, 3, name="atoms")  # sites 0, 1, 2
    illegal = Program([atoms])
    illegal.add(ops.LoadAtoms(1, 3))
    illegal.add(ops.CZ, (atoms[0], atoms[2]))  # sites 0 and 2: not neighbors
    with pytest.raises(BackendValidationError):
        AtomGridSimulator(grid_size=(1, 3))._lower_program(illegal)

    legal = Program([atoms])
    legal.add(ops.LoadAtoms(1, 3))
    legal.add(ops.Rearrange((2, 1)), (atoms[1], atoms[2]))  # swap: atoms[2] -> site 1
    legal.add(ops.CZ, (atoms[0], atoms[2]))  # now nearest neighbors -> legal
    AtomGridSimulator(grid_size=(1, 3))._lower_program(legal)  # must not raise


def test_rearrange_does_not_change_state():
    atoms = GridRegister(1, 2, name="atoms")  # 2 atoms on a 1x3 device

    def build(move):
        p = Program([atoms])
        p.add(ops.LoadAtoms(1, 2))
        p.add(ops.RY(0.7), atoms[0])
        p.add(ops.RX(1.1), atoms[1])
        if move:
            p.add(ops.Rearrange((2,)), atoms[0])  # move atoms[0] to a free site
        return p

    def sv(p):
        return (
            AtomGridSimulator(grid_size=(1, 3))
            .run(p, result_config={"counts": False, "final_state": True})
            .result().get_statevector()
        )

    assert np.allclose(sv(build(True)), sv(build(False)))


def test_atomic_swap_exchanges_sites_not_state():
    atoms = GridRegister(1, 2, name="atoms")

    def build(swap):
        p = Program([atoms])
        p.add(ops.LoadAtoms(1, 2))
        p.add(ops.RY(0.9), atoms[0])
        p.add(ops.RX(0.4), atoms[1])
        if swap:
            p.add(ops.Rearrange((1, 0)), (atoms[0], atoms[1]))  # atomic swap
        return p

    def sv(p):
        return (
            AtomGridSimulator(grid_size=(1, 2))
            .run(p, result_config={"counts": False, "final_state": True})
            .result().get_statevector()
        )

    # Swap exchanges trap sites, not state, and needs no temp site.
    assert np.allclose(sv(build(True)), sv(build(False)))


def test_atom_moved_outside_load_block_still_accepts_gates():
    atoms = GridRegister(1, 2, name="atoms")  # 2 atoms on a 1x3 device
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))               # load sites {0, 1}
    p.add(ops.Rearrange((2,)), atoms[0])     # move atoms[0] to site 2 (outside block)
    p.add(ops.RX(np.pi), atoms[0])           # must still execute
    p.measure(atoms[0], 0)
    counts = (
        AtomGridSimulator(grid_size=(1, 3))
        .run(p, shots=8, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"1": 8}                # RX(pi)|0> -> |1>, not dropped


def test_conditional_rearrange_rejected():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))
    p.measure(atoms[0], 0)
    p.add(ops.Rearrange((2,)), atoms[0], condition=(0, 1)) 
    with pytest.raises(BackendValidationError):
        AtomGridSimulator(grid_size=(1, 3))._lower_program(p)


def test_rearrange_non_injective_rejected():
    atoms = GridRegister(1, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.LoadAtoms(1, 3))
    p.add(ops.Rearrange((1,)), atoms[0])     # onto atoms[1]'s site -> collision
    with pytest.raises(BackendValidationError):
        AtomGridSimulator(grid_size=(1, 3))._lower_program(p)


def test_rearrange_target_site_off_device_rejected():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms])
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.Rearrange((9,)), atoms[0])     # site 9 does not exist on 1x3
    with pytest.raises(BackendValidationError):
        AtomGridSimulator(grid_size=(1, 3))._lower_program(p)


def test_rearrange_of_unloaded_operand_is_silently_ignored():
    atoms = GridRegister(1, 2, name="atoms")  # on 1x3 device
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 1))               # load site 0 only; atoms[1] unloaded
    p.add(ops.Rearrange((2,)), atoms[1])     # move the empty operand: must NOT raise
    p.add(ops.RX(np.pi), atoms[1])           # still dropped (unloaded)
    p.measure(atoms[1], 0)
    counts = (
        AtomGridSimulator(grid_size=(1, 3))
        .run(p, shots=4, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"0": 4}                # atoms[1] never got a gate -> reads 0


def test_refill_restores_a_lost_site():
    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=1.0), operation=ops.RY)   # loss only on RY
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.RY(0.0), atoms[0])          # identity, but triggers p=1 loss on atoms[0]
    p.add(ops.Refill, atoms[0])           # reload -> fresh |0>
    p.add(ops.RX(np.pi), atoms[0])        # RX (no loss) works on the refilled atom
    p.measure(atoms[0], 0)
    counts = (
        AtomGridSimulator(grid_size=(1, 2), noise=noise)
        .run(p, shots=8, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"1": 8}             # refilled |0> -> RX(pi) -> |1>, reads 1 (not erasure 2)


def test_refill_on_occupied_site_is_a_noop():
    atoms = GridRegister(1, 2, name="atoms")

    def build(with_refill):
        p = Program([atoms], 1)
        p.add(ops.LoadAtoms(1, 2))
        p.add(ops.RX(np.pi), atoms[0])    # atoms[0] -> |1>
        if with_refill:
            p.add(ops.Refill, atoms[0])   # occupied -> must be a no-op (must NOT reset to |0>)
        p.measure(atoms[0], 0)
        return p

    def counts(p):
        return (
            AtomGridSimulator(grid_size=(1, 2))
            .run(p, shots=8, simulation_config={"seed": 0})
            .result().get_counts()
        )

    assert counts(build(True)) == counts(build(False)) == {"1": 8}   


def test_refill_can_fill_a_never_loaded_site():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 1))            # load site 0 only; atoms[1] never loaded
    p.add(ops.Refill, atoms[1])          # fill the never-loaded site
    p.add(ops.RX(np.pi), atoms[1])       # usable -> proves static drop was narrowed
    p.measure(atoms[1], 0)
    counts = (
        AtomGridSimulator(grid_size=(1, 2))
        .run(p, shots=8, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"1": 8}            # refilled |0> -> RX(pi) -> |1>


def test_refill_loss_gives_loading_efficiency():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 1))            # load site 0; atoms[1] empty
    p.add(ops.Refill, atoms[1])          # try to load atoms[1]: succeeds ~60%
    p.measure(atoms[1], 0)              

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=0.4), operation=ops.Refill)
    counts = (
        AtomGridSimulator(grid_size=(1, 2), noise=noise)
        .run(p, shots=4000, simulation_config={"seed": 0})
        .result().get_counts()
    )
    total = sum(counts.values())
    assert 0.55 < counts.get("0", 0) / total < 0.65   # ~60% loaded


def test_rearrange_loss_ejects_the_moved_atom():
    atoms = GridRegister(1, 2, name="atoms")     # 1x3 device, site 2 free
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.Rearrange((2,)), atoms[0])         # move atoms[0]; loss fires on it 
    p.measure(atoms[0], 0)

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=1.0), operation=ops.Rearrange)
    counts = (
        AtomGridSimulator(grid_size=(1, 3), noise=noise)
        .run(p, shots=8, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"2": 8}                     # moved atom lost -> erasure


def test_rearrange_loss_spares_an_unmoved_atom():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.Rearrange((2,)), atoms[0])         # only atoms[0] moves
    p.measure(atoms[1], 0)                         # atoms[1] not moved -> not lost

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=1.0), operation=ops.Rearrange)
    counts = (
        AtomGridSimulator(grid_size=(1, 3), noise=noise)
        .run(p, shots=8, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"0": 8}


def test_rearrange_kraus_noise_applies_to_moved_atom():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.RX(np.pi), atoms[0])               # atoms[0] -> |1>
    p.add(ops.Rearrange((2,)), atoms[0])         # move; Depolarizing p=1 -> I/2
    p.measure(atoms[0], 0)

    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=1.0), operation=ops.Rearrange)   # full depolarize
    counts = (
        AtomGridSimulator(grid_size=(1, 3), noise=noise)
        .run(p, shots=2000, simulation_config={"seed": 0})
        .result().get_counts()
    )
    total = sum(counts.values())
    assert 0.4 < counts.get("1", 0) / total < 0.6   # fully mixed -> 50/50


def test_loss_rearrange_refill_compose():
    atoms = GridRegister(1, 2, name="atoms")
    p = Program([atoms], 1)
    p.add(ops.LoadAtoms(1, 2))
    p.add(ops.RY(0.0), atoms[0])                  # lose atoms[0]
    p.add(ops.Rearrange((2,)), atoms[1])          # move atoms[1] (independent)
    p.add(ops.Refill, atoms[0])                   # reload atoms[0] -> fresh |0>
    p.add(ops.RX(np.pi), atoms[0])                # usable again -> |1>
    p.measure(atoms[0], 0)

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=1.0), operation=ops.RY)
    counts = (
        AtomGridSimulator(grid_size=(1, 3), noise=noise)
        .run(p, shots=8, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"1": 8}     # loss + refill + gate compose; rearrange doesn't interfere