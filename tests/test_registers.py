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
    qubits = GridRegister(2, 3)
    assert isinstance(qubits, QuantumRegister)
    assert qubits.rows == 2
    assert qubits.cols == 3
    assert qubits.size == 6
    assert qubits.dim == 2


def test_grid_register_accepts_name_metadata_dim_by_keyword():
    qubits = GridRegister(2, 3, name="qubits", metadata={"k": 1}, dim=2)
    assert qubits.name == "qubits"
    assert qubits.metadata == {"k": 1}


def test_grid_register_metadata_is_copied_not_aliased():
    meta = {"k": 1}
    qubits = GridRegister(2, 2, metadata=meta)
    meta["k"] = 2
    assert qubits.metadata == {"k": 1}


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
    qubits = GridRegister(2, 3)
    with pytest.raises(Exception):
        qubits.rows = 5


def test_grid_register_scalar_indexing_is_row_major():
    qubits = GridRegister(2, 3)
    # (row, col) -> row * cols + col
    for row in range(2):
        for col in range(3):
            index = row * 3 + col
            ref = qubits[index]
            assert isinstance(ref, RegisterRef)
            assert ref.register is qubits
            assert ref.index == index


def test_grid_register_scalar_indexing_out_of_range_raises_indexerror():
    qubits = GridRegister(2, 3)
    with pytest.raises(IndexError):
        qubits[6]
    with pytest.raises(IndexError):
        qubits[-1]


def test_distinct_grid_registers_with_identical_fields_are_not_equal():
    assert GridRegister(2, 3) != GridRegister(2, 3)


def test_grid_registers_are_hashable_and_identity_based():
    qubits = GridRegister(2, 3)
    assert isinstance(hash(qubits), int)
    assert len({GridRegister(2, 3), GridRegister(2, 3)}) == 2
    assert len({qubits, qubits}) == 1


# --- GridRegister selection helpers -----------------------------------------


def test_all_returns_view_with_all_selector():
    qubits = GridRegister(2, 3)
    view = qubits.all()
    assert isinstance(view, RegisterView)
    assert view.register is qubits
    assert view.selector == AllSelector()


def test_row_returns_view_with_row_selector():
    qubits = GridRegister(2, 3)
    view = qubits.row(1)
    assert view.register is qubits
    assert view.selector == RowSelector(row=1)


def test_row_rejects_out_of_bounds_index():
    qubits = GridRegister(2, 3)
    with pytest.raises(IndexError):
        qubits.row(2)
    with pytest.raises(IndexError):
        qubits.row(-1)


def test_row_rejects_non_int_index():
    qubits = GridRegister(2, 3)
    with pytest.raises(TypeError):
        qubits.row(1.0)
    with pytest.raises(TypeError):
        qubits.row(True)


def test_column_returns_view_with_column_selector():
    qubits = GridRegister(2, 3)
    view = qubits.column(2)
    assert view.register is qubits
    assert view.selector == ColumnSelector(col=2)


def test_column_rejects_out_of_bounds_index():
    qubits = GridRegister(2, 3)
    with pytest.raises(IndexError):
        qubits.column(3)
    with pytest.raises(IndexError):
        qubits.column(-1)


def test_column_rejects_non_int_index():
    qubits = GridRegister(2, 3)
    with pytest.raises(TypeError):
        qubits.column(1.0)
    with pytest.raises(TypeError):
        qubits.column(True)


def test_block_returns_view_with_block_selector():
    qubits = GridRegister(4, 5)
    view = qubits.block(rows=(0, 2), cols=(1, 3))
    assert view.register is qubits
    assert view.selector == BlockSelector(rows=(0, 2), cols=(1, 3))


@pytest.mark.parametrize(
    "rows,cols",
    [
        ((0, 0), (0, 2)),  # start == stop
        ((2, 1), (0, 2)),  # start > stop
        ((0, 2), (0, 0)),
        ((0, 2), (2, 1)),
    ],
)
def test_block_rejects_malformed_ranges(rows, cols):
    qubits = GridRegister(4, 5)
    with pytest.raises(ValueError):
        qubits.block(rows=rows, cols=cols)


@pytest.mark.parametrize(
    "rows,cols",
    [
        ((-1, 2), (0, 2)),  # start < 0
        ((0, 5), (0, 2)),  # stop > limit
        ((0, 2), (-1, 2)),
        ((0, 2), (0, 6)),
    ],
)
def test_block_rejects_out_of_bounds_ranges_with_indexerror(rows, cols):
    # Out-of-bounds coordinates raise IndexError, matching row()/column().
    qubits = GridRegister(4, 5)
    with pytest.raises(IndexError):
        qubits.block(rows=rows, cols=cols)


def test_block_rejects_non_int_range_endpoints():
    qubits = GridRegister(4, 5)
    with pytest.raises(TypeError):
        qubits.block(rows=(0.0, 2), cols=(0, 2))
    with pytest.raises(TypeError):
        qubits.block(rows=(0, 2), cols=(0, 2.0))


def test_block_full_grid_is_valid():
    qubits = GridRegister(4, 5)
    view = qubits.block(rows=(0, 4), cols=(0, 5))
    assert view.selector == BlockSelector(rows=(0, 4), cols=(0, 5))


# --- RegisterView / selector immutability and hashability -------------------


def test_register_view_is_frozen():
    qubits = GridRegister(2, 3)
    view = qubits.all()
    with pytest.raises(Exception):
        view.selector = RowSelector(row=0)


def test_register_view_is_hashable():
    qubits = GridRegister(2, 3)
    assert isinstance(hash(qubits.all()), int)
    assert isinstance(hash(qubits.row(0)), int)
    assert isinstance(hash(qubits.column(0)), int)
    assert isinstance(hash(qubits.block(rows=(0, 1), cols=(0, 1))), int)


def test_register_view_usable_in_set_and_as_dict_key():
    qubits = GridRegister(2, 3)
    views = {qubits.all(), qubits.row(0), qubits.row(0), qubits.column(1)}
    assert len(views) == 3
    mapping = {qubits.row(0): "row0"}
    assert mapping[qubits.row(0)] == "row0"


def test_register_view_equal_views_on_same_register_compare_equal():
    qubits = GridRegister(2, 3)
    assert qubits.row(0) == qubits.row(0)
    assert qubits.all() == qubits.all()


def test_register_view_on_different_registers_are_not_equal():
    a = GridRegister(2, 3)
    b = GridRegister(2, 3)
    assert a.row(0) != b.row(0)


def test_register_view_different_selectors_are_not_equal():
    qubits = GridRegister(2, 3)
    assert qubits.row(0) != qubits.row(1)
    assert qubits.row(0) != qubits.column(0)
    assert qubits.all() != qubits.row(0)


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


def test_register_len_matches_size_and_iteration():
    reg = QuantumRegister(3, name="q")
    assert len(reg) == 3
    assert [ref.index for ref in reg] == [0, 1, 2]


def test_register_out_of_range_message_names_size():
    reg = QuantumRegister(2, name="q")
    with pytest.raises(IndexError, match="out of range for size 2"):
        reg[5]
    with pytest.raises(IndexError, match="negative indexing is not supported"):
        reg[-1]


def test_grid_out_of_range_coordinates_raise_indexerror_consistently():
    grid = GridRegister(2, 3, name="g")
    with pytest.raises(IndexError):
        grid.row(5)
    with pytest.raises(IndexError):
        grid.column(9)
    with pytest.raises(IndexError):
        grid.block((0, 5), (0, 1))
    with pytest.raises(IndexError):
        grid.block((0, 1), (-1, 2))
    # malformed (empty/inverted) ranges are still ValueError
    with pytest.raises(ValueError, match="start < stop"):
        grid.block((1, 1), (0, 1))
