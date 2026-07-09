"""Fake 4x4 superconducting-grid backend for compiler prototyping.

Fixed 16-qubit device, row-major numbered:

.. code-block:: text

    0   1   2   3
    4   5   6   7
    8   9  10  11
    12 13  14  15

Native gate set is exactly ``RZ``, ``SX`` (single-qubit, any of the 16
labels), and ``CZ`` (nearest-neighbor edges only, both directions stored).
This is a prototype execution target for the compiler group, not a realistic
device model: no routing, no timing, no noise. See
``docs/superpowers/specs/2026-07-09-fatqat-target-aware-implementation-map-and-4x4-fake-superconducting-backend-design.md``.

The native-gate-set restriction applies to unitary operations only.
Measurement and reset are resolved by `StateVectorBackend._lower` before any
implementation-map lookup happens (see the `isinstance` dispatch there), so
this backend accepts them on any of the 16 qubits regardless of the
target-aware map's contents.
"""

from __future__ import annotations

from typing import Any

from .. import operations as ops
from ..errors import BackendValidationError
from ..implementation import (
    MatrixImplementation,
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
from ..layout import ResourceLayout
from ..operations import Operation
from ..program import Program
from .statevector_backend import StateVectorBackend

GRID_ROWS = 4
GRID_COLS = 4
N_QUBITS = GRID_ROWS * GRID_COLS


def _nearest_neighbor_edges() -> tuple[tuple[int, int], ...]:
    """Return directed nearest-neighbor edges for a row-major 4x4 grid.

    Both directions of every edge are included (e.g. `(0, 1)` and `(1, 0)`),
    per the design's "keep lookup simple, never reorder targets" rule.
    """
    edges: list[tuple[int, int]] = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            q = row * GRID_COLS + col
            if col + 1 < GRID_COLS:
                right = q + 1
                edges.extend(((q, right), (right, q)))
            if row + 1 < GRID_ROWS:
                down = q + GRID_COLS
                edges.extend(((q, down), (down, q)))
    return tuple(edges)


def _require_rule(
    implementation_map: MatrixImplementationMap,
    op: Operation,
) -> MatrixImplementation:
    rule = implementation_map.get(op)
    if rule is None:
        raise RuntimeError(f"default matrix implementation missing for {op!r}")
    return rule


def fake_superconducting_4x4_implementation_map() -> MatrixImplementationMap:
    """Build the native gate map for the fake 4x4 superconducting backend.

    `RZ` and `SX` are legal on all 16 qubit labels; `CZ` is legal only on
    nearest-neighbor grid edges, both directions. Every other operation
    family (including `CX`) has no entry and is therefore unsupported.
    """
    defaults = default_matrix_implementation_map()
    rz_rule = _require_rule(defaults, ops.RZ)
    sx_rule = _require_rule(defaults, ops.SX)
    cz_rule = _require_rule(defaults, ops.CZ)

    m = MatrixImplementationMap()
    for q in range(N_QUBITS):
        m.register_for(ops.RZ, (q,), rz_rule)
        m.register_for(ops.SX, (q,), sx_rule)
    for edge in _nearest_neighbor_edges():
        m.register_for(ops.CZ, edge, cz_rule)
    return m


class FakeSuperconducting4x4Backend(StateVectorBackend):
    """Statevector backend constrained to a fake 4x4 superconducting target.

    A thin `StateVectorBackend` specialization: same execution engine, same
    `Result`/`Job` semantics. The only differences are a fixed 16-qubit
    device, a fixed native gate set (`RZ`, `SX`, nearest-neighbor `CZ`), and
    rejecting programs that do not fit that device shape (wrong qubit count,
    or any non-qubit-dimension register).
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Create a fake 4x4 superconducting backend.

        Args:
            options: Same execution-strategy options as `StateVectorBackend`
                (``max_workers``, ``parallel_mode``). The implementation map
                is fixed to `fake_superconducting_4x4_implementation_map()`
                and cannot be overridden.
        """
        super().__init__(
            options=options,
            implementation_map=fake_superconducting_4x4_implementation_map(),
        )

    @property
    def implementation_map(self) -> MatrixImplementationMap:
        """Return a copy of the compiler-facing target-aware implementation map."""
        return self._impl_map.copy()

    def resolve_layout(self, program: Program) -> ResourceLayout:
        """Build the flat layout, then reject any shape the fake device can't run.

        Raises:
            BackendValidationError: If the program does not declare exactly
                16 qubits, or declares any non-qubit-dimension (`dim != 2`)
                register. This prototype does not perform qubit mapping, so
                a smaller or differently-shaped program is rejected rather
                than silently mapped onto a subset of the device.
        """
        layout = super().resolve_layout(program)
        if layout.n_subsystems != N_QUBITS:
            raise BackendValidationError(
                f"FakeSuperconducting4x4Backend requires exactly 16 qubits, "
                f"got {layout.n_subsystems}"
            )
        if any(dim != 2 for dim in layout.system_dims):
            raise BackendValidationError(
                "FakeSuperconducting4x4Backend only supports qubit dimensions"
            )
        return layout
