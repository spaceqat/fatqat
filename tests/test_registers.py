"""Tests register construction, indexing, metadata copying, and immutability."""

import pytest

from fatqat.registers import (
    AllSelector,
    BlockSelector,
    ClassicalRegister,
    ColumnSelector,
    GridRegister,
    QuantumRegister,
    RegisterRef,
    RegisterView,
    RowSelector,
)


def test_getitem_returns_registerref():
    qr = QuantumRegister(3, name="q")
    ref = qr[1]
    assert isinstance(ref, RegisterRef)
    assert ref.register is qr
    assert ref.index == 1


def test_getitem_out_of_range_raises_indexerror():
    qr = QuantumRegister(2)
    with pytest.raises(IndexError):
        qr[2]
    with pytest.raises(IndexError):
        qr[-1]


def test_size_first_construction_and_keyword_name():
    cr = ClassicalRegister(4, name="c")
    assert cr.size == 4
    assert cr.name == "c"


def test_non_positive_size_rejected():
    with pytest.raises(ValueError):
        QuantumRegister(0)
    with pytest.raises(ValueError):
        ClassicalRegister(-1)


def test_registers_are_frozen():
    qr = QuantumRegister(1)
    with pytest.raises(Exception):
        qr.size = 5


def test_register_metadata_is_copied_not_aliased():
    meta = {"k": 1}
    qr = QuantumRegister(1, metadata=meta)
    meta["k"] = 2
    assert qr.metadata == {"k": 1}


def test_dim_defaults_to_two():
    assert QuantumRegister(3).dim == 2
    assert ClassicalRegister(2).dim == 2


def test_dim_can_be_set():
    qr = QuantumRegister(3, dim=3)
    assert qr.dim == 3
    assert qr[0].register.dim == 3


def test_dim_below_two_rejected():
    with pytest.raises(ValueError):
        QuantumRegister(2, dim=1)
    with pytest.raises(ValueError):
        QuantumRegister(2, dim=0)
    with pytest.raises(ValueError):
        ClassicalRegister(2, dim=-3)


def test_dim_non_int_rejected():
    with pytest.raises(TypeError):
        QuantumRegister(2, dim=2.0)
    with pytest.raises(TypeError):
        QuantumRegister(2, dim=True)


def test_distinct_registers_with_identical_fields_are_not_equal():
    assert QuantumRegister(2, name="q") != QuantumRegister(2, name="q")
    assert ClassicalRegister(2, name="c") != ClassicalRegister(2, name="c")


def test_registers_are_hashable_despite_metadata():
    assert isinstance(hash(QuantumRegister(1, metadata={"k": 1})), int)
    assert isinstance(hash(ClassicalRegister(1, metadata={"k": 1})), int)


def test_distinct_registers_do_not_collapse_in_a_set():
    assert len({QuantumRegister(1), QuantumRegister(1)}) == 2


def test_registerref_compares_by_register_identity_and_index():
    a = QuantumRegister(2, name="q")
    b = QuantumRegister(2, name="q")
    assert a[0] == a[0]
    assert a[0] != a[1]
    assert a[0] != b[0]


def test_registerref_is_usable_as_a_dict_key():
    a = QuantumRegister(2, name="q")
    b = QuantumRegister(2, name="q")
    offsets = {a[0]: "a0", b[0]: "b0"}
    assert offsets[a[0]] == "a0"
    assert offsets[b[0]] == "b0"
    assert len(offsets) == 2


# --- GridRegister -----------------------------------------------------------


def test_grid_register_is_quantum_register_with_derived_size():
    atoms = GridRegister(2, 3)
    assert isinstance(atoms, QuantumRegister)
    assert atoms.rows == 2
    assert atoms.cols == 3
    assert atoms.size == 6
    assert atoms.dim == 2


def test_grid_register_accepts_name_metadata_dim_by_keyword():
    atoms = GridRegister(2, 3, name="atoms", metadata={"k": 1}, dim=2)
    assert atoms.name == "atoms"
    assert atoms.metadata == {"k": 1}


def test_grid_register_metadata_is_copied_not_aliased():
    meta = {"k": 1}
    atoms = GridRegister(2, 2, metadata=meta)
    meta["k"] = 2
    assert atoms.metadata == {"k": 1}


@pytest.mark.parametrize("rows,cols", [(0, 3), (-1, 3), (3, 0), (3, -1)])
def test_grid_register_non_positive_dimensions_rejected(rows, cols):
    with pytest.raises(ValueError):
        GridRegister(rows, cols)


@pytest.mark.parametrize("rows,cols", [(2.0, 3), (True, 3), (2, 3.0), (2, True)])
def test_grid_register_non_int_dimensions_rejected(rows, cols):
    with pytest.raises(TypeError):
        GridRegister(rows, cols)


def test_grid_register_dim_follows_quantum_register_rules():
    with pytest.raises(ValueError):
        GridRegister(2, 3, dim=1)
    with pytest.raises(TypeError):
        GridRegister(2, 3, dim=2.0)
    qutrits = GridRegister(2, 3, dim=3)
    assert qutrits.dim == 3


def test_grid_register_is_frozen():
    atoms = GridRegister(2, 3)
    with pytest.raises(Exception):
        atoms.rows = 5


def test_grid_register_scalar_indexing_is_row_major():
    atoms = GridRegister(2, 3)
    # (row, col) -> row * cols + col
    for row in range(2):
        for col in range(3):
            index = row * 3 + col
            ref = atoms[index]
            assert isinstance(ref, RegisterRef)
            assert ref.register is atoms
            assert ref.index == index


def test_grid_register_scalar_indexing_out_of_range_raises_indexerror():
    atoms = GridRegister(2, 3)
    with pytest.raises(IndexError):
        atoms[6]
    with pytest.raises(IndexError):
        atoms[-1]


def test_distinct_grid_registers_with_identical_fields_are_not_equal():
    assert GridRegister(2, 3) != GridRegister(2, 3)


def test_grid_registers_are_hashable_and_identity_based():
    atoms = GridRegister(2, 3)
    assert isinstance(hash(atoms), int)
    assert len({GridRegister(2, 3), GridRegister(2, 3)}) == 2
    assert len({atoms, atoms}) == 1


# --- GridRegister selection helpers -----------------------------------------


def test_all_returns_view_with_all_selector():
    atoms = GridRegister(2, 3)
    view = atoms.all()
    assert isinstance(view, RegisterView)
    assert view.register is atoms
    assert view.selector == AllSelector()


def test_row_returns_view_with_row_selector():
    atoms = GridRegister(2, 3)
    view = atoms.row(1)
    assert view.register is atoms
    assert view.selector == RowSelector(row=1)


def test_row_rejects_out_of_bounds_index():
    atoms = GridRegister(2, 3)
    with pytest.raises(IndexError):
        atoms.row(2)
    with pytest.raises(IndexError):
        atoms.row(-1)


def test_row_rejects_non_int_index():
    atoms = GridRegister(2, 3)
    with pytest.raises(TypeError):
        atoms.row(1.0)
    with pytest.raises(TypeError):
        atoms.row(True)


def test_column_returns_view_with_column_selector():
    atoms = GridRegister(2, 3)
    view = atoms.column(2)
    assert view.register is atoms
    assert view.selector == ColumnSelector(col=2)


def test_column_rejects_out_of_bounds_index():
    atoms = GridRegister(2, 3)
    with pytest.raises(IndexError):
        atoms.column(3)
    with pytest.raises(IndexError):
        atoms.column(-1)


def test_column_rejects_non_int_index():
    atoms = GridRegister(2, 3)
    with pytest.raises(TypeError):
        atoms.column(1.0)
    with pytest.raises(TypeError):
        atoms.column(True)


def test_block_returns_view_with_block_selector():
    atoms = GridRegister(4, 5)
    view = atoms.block(rows=(0, 2), cols=(1, 3))
    assert view.register is atoms
    assert view.selector == BlockSelector(rows=(0, 2), cols=(1, 3))


@pytest.mark.parametrize(
    "rows,cols",
    [
        ((0, 0), (0, 2)),  # start == stop
        ((2, 1), (0, 2)),  # start > stop
        ((-1, 2), (0, 2)),  # start < 0
        ((0, 5), (0, 2)),  # stop > limit
        ((0, 2), (0, 0)),
        ((0, 2), (2, 1)),
        ((0, 2), (-1, 2)),
        ((0, 2), (0, 6)),
    ],
)
def test_block_rejects_invalid_half_open_ranges(rows, cols):
    atoms = GridRegister(4, 5)
    with pytest.raises(ValueError):
        atoms.block(rows=rows, cols=cols)


def test_block_rejects_non_int_range_endpoints():
    atoms = GridRegister(4, 5)
    with pytest.raises(TypeError):
        atoms.block(rows=(0.0, 2), cols=(0, 2))
    with pytest.raises(TypeError):
        atoms.block(rows=(0, 2), cols=(0, 2.0))


def test_block_full_grid_is_valid():
    atoms = GridRegister(4, 5)
    view = atoms.block(rows=(0, 4), cols=(0, 5))
    assert view.selector == BlockSelector(rows=(0, 4), cols=(0, 5))


# --- RegisterView / selector immutability and hashability -------------------


def test_register_view_is_frozen():
    atoms = GridRegister(2, 3)
    view = atoms.all()
    with pytest.raises(Exception):
        view.selector = RowSelector(row=0)


def test_register_view_is_hashable():
    atoms = GridRegister(2, 3)
    assert isinstance(hash(atoms.all()), int)
    assert isinstance(hash(atoms.row(0)), int)
    assert isinstance(hash(atoms.column(0)), int)
    assert isinstance(hash(atoms.block(rows=(0, 1), cols=(0, 1))), int)


def test_register_view_usable_in_set_and_as_dict_key():
    atoms = GridRegister(2, 3)
    views = {atoms.all(), atoms.row(0), atoms.row(0), atoms.column(1)}
    assert len(views) == 3
    mapping = {atoms.row(0): "row0"}
    assert mapping[atoms.row(0)] == "row0"


def test_register_view_equal_views_on_same_register_compare_equal():
    atoms = GridRegister(2, 3)
    assert atoms.row(0) == atoms.row(0)
    assert atoms.all() == atoms.all()


def test_register_view_on_different_registers_are_not_equal():
    a = GridRegister(2, 3)
    b = GridRegister(2, 3)
    assert a.row(0) != b.row(0)


def test_register_view_different_selectors_are_not_equal():
    atoms = GridRegister(2, 3)
    assert atoms.row(0) != atoms.row(1)
    assert atoms.row(0) != atoms.column(0)
    assert atoms.all() != atoms.row(0)


def test_selectors_are_hashable():
    for selector in (
        AllSelector(),
        RowSelector(row=1),
        ColumnSelector(col=1),
        BlockSelector(rows=(0, 1), cols=(0, 1)),
    ):
        assert isinstance(hash(selector), int)


def test_selectors_are_frozen():
    with pytest.raises(Exception):
        RowSelector(row=1).row = 2
    with pytest.raises(Exception):
        ColumnSelector(col=1).col = 2
    with pytest.raises(Exception):
        BlockSelector(rows=(0, 1), cols=(0, 1)).rows = (0, 2)


def test_selectors_compare_by_value():
    assert RowSelector(row=1) == RowSelector(row=1)
    assert RowSelector(row=1) != RowSelector(row=2)
    assert AllSelector() == AllSelector()
    assert ColumnSelector(col=1) == ColumnSelector(col=1)
    assert BlockSelector(rows=(0, 1), cols=(0, 2)) == BlockSelector(
        rows=(0, 1), cols=(0, 2)
    )
