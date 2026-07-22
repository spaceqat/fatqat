"""Tests the fake atom grid backend's grid binding, device-shape, and native
gate-set constraints.
"""

import numpy as np
import pytest

from fatqat import operations as ops
from fatqat.backends import FakeAtomGridBackend, SimulatorBackend
from fatqat.backends.fake_atom_grid import fake_atom_grid_implementation_map
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister

# --- constructor validation / default shape -----------------------------------


def test_default_shape_is_4x5():
    backend = FakeAtomGridBackend()
    p_fits = Program(20)
    p_fits.add(ops.RX(0.1), 0)
    # 4x5 = 20 qubits fits exactly.
    layout = backend.resolve_layout(p_fits)
    assert layout.n_subsystems == 20

    p_too_big = Program(21)
    p_too_big.add(ops.RX(0.1), 0)
    with pytest.raises(BackendValidationError):
        backend.resolve_layout(p_too_big)


def test_rejects_non_int_rows():
    with pytest.raises(TypeError):
        FakeAtomGridBackend(rows=2.0, cols=3)


def test_rejects_non_int_cols():
    with pytest.raises(TypeError):
        FakeAtomGridBackend(rows=2, cols="3")


def test_rejects_bool_rows():
    with pytest.raises(TypeError):
        FakeAtomGridBackend(rows=True, cols=3)


def test_rejects_bool_cols():
    with pytest.raises(TypeError):
        FakeAtomGridBackend(rows=2, cols=False)


def test_rejects_zero_rows():
    with pytest.raises(ValueError):
        FakeAtomGridBackend(rows=0, cols=3)


def test_rejects_negative_cols():
    with pytest.raises(ValueError):
        FakeAtomGridBackend(rows=2, cols=-1)


# --- 2x3-on-4x5 acceptance example ---------------------------------------------


def test_2x3_grid_binds_top_left_on_default_4x5_backend():
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    p.add(ops.RX(0.1), atoms.all())

    backend = FakeAtomGridBackend()  # default 4x5
    layout = backend.resolve_layout(p)
    resources = backend._build_qubit_resource_map(p, layout)

    assert tuple(resources[atoms[i]].device_label for i in range(6)) == (
        0,
        1,
        2,
        5,
        6,
        7,
    )
    assert tuple(resources[atoms[i]].engine_index for i in range(6)) == tuple(range(6))


def test_scalar_grid_ref_uses_grid_binder_device_label_not_identity():
    # atoms[3] is frontend (row=1, col=0) on a 2x3 grid: engine index 3 (flat
    # layout order), but device label must come from the *backend*'s column
    # count (5), i.e. 1*5+0 = 5, not the identity value 3.
    atoms = GridRegister(2, 3, name="atoms")
    p = Program([atoms])
    backend = FakeAtomGridBackend()  # default 4x5
    layout = backend.resolve_layout(p)
    ref = atoms[3]
    bound = backend._build_qubit_resource_map(p, layout)[ref]
    assert bound.engine_index == 3
    assert bound.device_label == 5


# --- sole-register / fit / multiplicity rules ----------------------------------


def test_rejects_grid_register_combined_with_other_quantum_register():
    atoms = GridRegister(2, 2, name="atoms")
    other = QuantumRegister(2, name="q")
    p = Program([atoms, other])
    backend = FakeAtomGridBackend()
    with pytest.raises(BackendValidationError):
        backend.resolve_layout(p)


def test_rejects_two_grid_registers():
    atoms1 = GridRegister(2, 2, name="a1")
    atoms2 = GridRegister(2, 2, name="a2")
    p = Program([atoms1, atoms2])
    backend = FakeAtomGridBackend()
    with pytest.raises(BackendValidationError):
        backend.resolve_layout(p)


def test_rejects_grid_that_does_not_fit_backend_even_with_enough_total_capacity():
    # 5x4 = 20 members, same total capacity as the default 4x5 backend, but
    # rows=5 exceeds the backend's 4 rows: must be rejected on a per-axis
    # basis, not just total member count.
    atoms = GridRegister(5, 4, name="atoms")
    p = Program([atoms])
    backend = FakeAtomGridBackend()
    with pytest.raises(BackendValidationError):
        backend.resolve_layout(p)


def test_resolve_layout_rejects_capacity_for_scalar_only_program():
    p = Program(21)
    backend = FakeAtomGridBackend()  # capacity 20
    with pytest.raises(BackendValidationError):
        backend.resolve_layout(p)


def test_resolve_layout_rejects_non_qubit_dimension_for_scalar_only_program():
    p = Program([QuantumRegister(4, dim=3)])
    backend = FakeAtomGridBackend()
    with pytest.raises(BackendValidationError, match="qubit dimensions"):
        backend.resolve_layout(p)


def test_scalar_only_program_uses_identity_binding():
    p = Program(3)
    backend = FakeAtomGridBackend()
    layout = backend.resolve_layout(p)
    ref = p.qreg[0][2]
    bound = backend._build_qubit_resource_map(p, layout)[ref]
    assert bound.engine_index == bound.device_label == 2


# --- implementation_map capability API -----------------------------------------


def test_implementation_map_exposes_five_native_families():
    m = FakeAtomGridBackend().implementation_map
    assert m.supports(ops.RX) and not m.device_operands_for(ops.RX)
    assert m.supports(ops.RY) and not m.device_operands_for(ops.RY)
    assert m.supports(ops.RZ) and not m.device_operands_for(ops.RZ)
    assert m.supports(ops.CX)
    assert m.supports(ops.CZ)


def test_implementation_map_cx_accepts_neighbor_pairs_both_directions():
    m = FakeAtomGridBackend().implementation_map  # default 4x5, backend cols=5
    assert (0, 1) in m.device_operands_for(ops.CX)
    assert (1, 0) in m.device_operands_for(ops.CX)
    assert (0, 5) in m.device_operands_for(ops.CX)
    assert (5, 0) in m.device_operands_for(ops.CX)


def test_implementation_map_cz_accepts_neighbor_pairs_both_directions():
    m = FakeAtomGridBackend().implementation_map
    assert (0, 1) in m.device_operands_for(ops.CZ)
    assert (1, 0) in m.device_operands_for(ops.CZ)
    assert (0, 5) in m.device_operands_for(ops.CZ)
    assert (5, 0) in m.device_operands_for(ops.CZ)


def test_implementation_map_rejects_non_neighbor_pairs():
    m = FakeAtomGridBackend().implementation_map
    assert (0, 2) not in m.device_operands_for(ops.CX)
    assert (0, 6) not in m.device_operands_for(ops.CX)
    assert (0, 2) not in m.device_operands_for(ops.CZ)


def test_fake_atom_grid_implementation_map_cx_has_no_class_keyed_rule():
    m = fake_atom_grid_implementation_map(4, 5)
    assert m.implementation_for(ops.CX) is None
    assert m.implementation_for(ops.CZ) is None


# --- resource-map behavior -----------------------------------------------------


def test_resource_map_contains_all_scalar_refs_from_the_program_registers():
    atoms = GridRegister(2, 2, name="atoms")
    p = Program([atoms])
    backend = FakeAtomGridBackend()
    layout = backend.resolve_layout(p)
    resources = backend._build_qubit_resource_map(p, layout)
    assert set(resources) == set(atoms[index] for index in range(4))


# --- integration: numeric equivalence to manual scalar circuits ----------------


def test_viewed_rotation_matches_manual_scalar_sequence():
    atoms = GridRegister(1, 3, name="atoms")
    grid_p = Program([atoms])
    grid_p.add(ops.RY(0.7), atoms.row(0))
    grid_sv = (
        FakeAtomGridBackend()
        .run(grid_p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(3)
    for i in range(3):
        manual_p.add(ops.RY(0.7), i)
    manual_sv = (
        SimulatorBackend()
        .run(manual_p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_paired_cx_view_matches_manual_scalar_sequence():
    # 2x2 grid on the default 4x5 backend: engine indices 0,1,2,3; device
    # labels 0,1,5,6 (backend cols=5). row(0) vs row(1) zips (atoms0,atoms2)
    # then (atoms1,atoms3) -> device pairs (0,5) and (1,6), both legal
    # vertical neighbor edges - even though the engine-index pairs (0,2) and
    # (1,3) are NOT adjacent under an identity mapping. This is the crux of
    # the task: device labels, not engine indices, drive gate legality.
    atoms = GridRegister(2, 2, name="atoms")
    grid_p = Program([atoms])
    grid_p.add(ops.CX, (atoms.row(0), atoms.row(1)))
    grid_sv = (
        FakeAtomGridBackend()
        .run(grid_p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(4)
    manual_p.add(ops.CX, (0, 2))
    manual_p.add(ops.CX, (1, 3))
    manual_sv = (
        SimulatorBackend()
        .run(manual_p, result_config={"counts": False, "statevector": True})
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
    grid_p.add(ops.RY(0.7), atoms.column(1))
    grid_sv = (
        FakeAtomGridBackend()
        .run(grid_p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(6)
    for i in (1, 4):
        manual_p.add(ops.RY(0.7), i)
    manual_sv = (
        SimulatorBackend()
        .run(manual_p, result_config={"counts": False, "statevector": True})
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
    grid_p.add(ops.RY(0.7), atoms.block(rows=(0, 2), cols=(1, 3)))
    grid_sv = (
        FakeAtomGridBackend()
        .run(grid_p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(6)
    for i in (1, 2, 4, 5):
        manual_p.add(ops.RY(0.7), i)
    manual_sv = (
        SimulatorBackend()
        .run(manual_p, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    assert np.allclose(grid_sv, manual_sv)


def test_non_neighbor_pair_rejects():
    atoms = GridRegister(2, 3, name="atoms")  # device labels row0:0,1,2 row1:5,6,7
    p = Program([atoms])
    p.add(ops.CX, (atoms[0], atoms[5]))  # labels 0 and 7: not adjacent

    with pytest.raises(UnsupportedOperationError) as excinfo:
        FakeAtomGridBackend().run(
            p, result_config={"counts": False, "statevector": True}
        )
    assert isinstance(excinfo.value, BackendValidationError)


def test_condition_on_viewed_instruction_propagates_end_to_end():
    atoms = GridRegister(1, 2, name="atoms")

    # Condition true (default clbit value is 0, condition checks == 0):
    # matches an unconditioned manual RX(pi) on both scalar qubits.
    grid_true = Program([atoms], 2)
    grid_true.add(ops.RX(np.pi), atoms.all(), condition=(grid_true.clreg[0][0], 0))
    sv_true = (
        FakeAtomGridBackend()
        .run(grid_true, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    manual_true = Program(2)
    manual_true.add(ops.RX(np.pi), 0)
    manual_true.add(ops.RX(np.pi), 1)
    sv_manual_true = (
        SimulatorBackend()
        .run(manual_true, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(sv_true, sv_manual_true)

    # Condition false (clbit compared to 1, but default is 0): must not fire,
    # state stays |00>.
    grid_false = Program([atoms], 2)
    grid_false.add(ops.RX(np.pi), atoms.all(), condition=(grid_false.clreg[0][0], 1))
    sv_false = (
        FakeAtomGridBackend()
        .run(grid_false, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )
    expected_false = np.zeros(4, dtype=complex)
    expected_false[0] = 1.0
    assert np.allclose(sv_false, expected_false)

    # And the two runs must actually differ - proof the condition mattered.
    assert not np.allclose(sv_true, sv_false)


def test_grid_register_export_is_backend_public():
    # FakeAtomGridBackend is publicly exported; implementation details are not.
    import fatqat.backends as backends_pkg

    assert "FakeAtomGridBackend" in backends_pkg.__all__
    assert not hasattr(backends_pkg, "_build_qubit_resource_map")
