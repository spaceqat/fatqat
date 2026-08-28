"""Superconducting hardware-profile simulators.

The IBM- and Google-style profiles use configurable row-major grids and
nearest-neighbour connectivity. They validate programs already written in
their native gate sets; neither profile routes, schedules, or models a named
device. Both are ideal unless a noise model is supplied.
"""

from __future__ import annotations

import numpy as np

from .. import operations as ops
from ..errors import BackendValidationError
from ..implementation import (
    MatrixImplementationMap,
    MatrixImplementation,
    default_matrix_implementation_map,
)
from ..noise import Depolarizing, NoiseModel, ReadoutConfusion, ThermalRelaxation
from ..operations import Operation
from ..program import Program
from ..registers import GridRegister, RegisterRef
from ..resource_layout import DeviceOperand, ResourceLayout
from .._backends.backend_utils import _validate_grid_size
from .simulator import Simulator

DEFAULT_ROWS = 4
DEFAULT_COLS = 4
DEFAULT_GRID_SIZE = (DEFAULT_ROWS, DEFAULT_COLS)

# --- fake calibration profile (the facts a real device would measure) ---
# Deliberately simple uniform numbers in realistic superconducting ranges;
# T2 stays within its physical bound T2 <= 2*T1, and readout is slightly
# asymmetric (reporting 1 for a true 0 is rarer than the reverse), matching
# the usual superconducting readout skew.
_T1 = 60e-6  # seconds
_T2 = 48e-6
_THERMAL_RELAXATION = ThermalRelaxation(t1=_T1, t2=_T2)
_SX_DURATION = 20e-9  # IBM-style RZ is virtual (zero duration -> no noise)
_ROTATION_DURATION = 20e-9  # Google-style RX/RY/RZ: all physical rotations
_CZ_DURATION = 50e-9
_ISWAP_DURATION = 30e-9
_CZ_DEPOLARIZING_P = 0.001
_ISWAP_DEPOLARIZING_P = 0.001
_READOUT_P01 = 0.02  # P(report 1 | true 0)
_READOUT_P10 = 0.04  # P(report 0 | true 1)


def _nearest_neighbor_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """Return directed nearest-neighbor edges for a row-major grid.

    Both directions of every edge are included (e.g. `(0, 1)` and `(1, 0)`),
    per the design's "keep lookup simple, never reorder targets" rule.
    """
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            q = row * cols + col
            if col + 1 < cols:
                right = q + 1
                edges.extend(((q, right), (right, q)))
            if row + 1 < rows:
                down = q + cols
                edges.extend(((q, down), (down, q)))
    return tuple(edges)


def _require_rule(
    implementation_map: MatrixImplementationMap,
    op: Operation,
) -> MatrixImplementation:
    rule = implementation_map.implementation_for(op)
    if rule is None:
        raise RuntimeError(f"default matrix implementation missing for {op!r}")
    return rule


class _SCQubitSimulator(Simulator):
    """Shared shape and resource-mapping logic for fake superconducting backends.

    Not part of the public API. `SCQubitIBMSimulator` and
    `SCQubitGoogleSimulator` both subclass this for their configurable device
    shape, GridRegister-aware resource mapping, and the
    `implementation_map` introspection property; each supplies its own
    native-gate implementation map and `default_noise_model`.
    """

    def __init__(
        self,
        implementation_map: MatrixImplementationMap,
        *,
        rows: int,
        cols: int,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        # rows/cols arrive pre-validated: each subclass's __init__ validates
        # the public grid_size tuple once, before building the implementation
        # map from the same shape.
        self._rows, self._cols = rows, cols
        super().__init__(
            method=method,
            runtime=runtime,
            implementation_map=implementation_map,
            noise=noise,
        )

    @property
    def implementation_map(self) -> MatrixImplementationMap:
        """Return the native operation map.

        ``supported_operations()`` lists native operation families.
        ``device_operands_for(operation)`` lists ordered device tuples for a
        connectivity-limited gate and is empty for a uniformly available gate.
        """
        return self._impl_map.copy()

    def _legal_device_operands(
        self, program: Program, resource_layout: ResourceLayout
    ) -> frozenset[DeviceOperand]:
        return frozenset(range(self._rows * self._cols))

    def _physical_dimension(
        self, device_operand: DeviceOperand, resource_layout: ResourceLayout
    ) -> int:
        return 2

    def _default_resource_layout(self, program: Program) -> ResourceLayout:
        """Reject any shape the fake device can't run, then map onto it.

        Applies equally to a scalar-only program with no `GridRegister`:
        total qubit count and per-subsystem dimension are checked regardless
        of register structure. A program's sole `GridRegister` (if any) then
        binds top-left onto the device: frontend `(row, col)` maps to device
        label `row * self._cols + col`. A scalar-only program (no
        `GridRegister`) delegates to the base class's generic
        declaration-order identity mapping, so an N-qubit program always
        maps onto physical qubits `0..N-1`.

        Raises:
            BackendValidationError: If the program declares more than this
                backend's capacity; any non-qubit-dimension (`dim != 2`) register; more
                than one `GridRegister`; a `GridRegister` combined with any
                other quantum register; or a `GridRegister` whose shape does
                not fit the device's, axis by axis.
        """
        name = type(self).__name__
        n_subsystems = sum(register.size for register in program.quantum_registers)
        capacity = self._rows * self._cols
        if n_subsystems > capacity:
            raise BackendValidationError(
                f"{name} supports at most {capacity} qubits on its "
                f"{self._rows}x{self._cols} device, got {n_subsystems}"
            )
        dims = (
            register.dim
            for register in program.quantum_registers
            for _ in range(register.size)
        )
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(f"{name} only supports qubit dimensions")
        grid_registers = [
            r for r in program.quantum_registers if isinstance(r, GridRegister)
        ]
        if len(grid_registers) > 1:
            raise BackendValidationError(
                f"{name} accepts at most one GridRegister per program, "
                f"got {len(grid_registers)}"
            )
        if not grid_registers:
            return super()._default_resource_layout(program)

        grid = grid_registers[0]
        if len(program.quantum_registers) != 1:
            raise BackendValidationError(
                f"{name} rejects a GridRegister combined with any other "
                "quantum register"
            )
        if grid.rows > self._rows or grid.cols > self._cols:
            raise BackendValidationError(
                f"grid register ({grid.rows}x{grid.cols}) does not fit "
                f"{name}'s ({self._rows}x{self._cols}) device shape"
            )
        labels: dict[RegisterRef, int] = {}
        for index in range(grid.size):
            row, col = divmod(index, grid.cols)
            labels[grid[index]] = row * self._cols + col
        return ResourceLayout(labels)


# --- IBM-style backend: X, SX, RZ, CZ --------------------------------------


def fake_superconducting_ibm_implementation_map(
    rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
) -> MatrixImplementationMap:
    """Build the native gate map for a `rows x cols` fake IBM-style superconducting backend.

    `X`, `SX`, and `RZ` are legal on any qubit label (registered uniformly
    via `add`); `CZ` is legal only on nearest-neighbor grid edges, both
    directions (added with explicit `device_operands`, one call per edge).
    Every other operation family (including `CX`) has no entry and is
    therefore unsupported.
    """
    rows, cols = _validate_grid_size((rows, cols))
    defaults = default_matrix_implementation_map()
    x_rule = _require_rule(defaults, ops.X)
    rz_rule = _require_rule(defaults, ops.RZ)
    sx_rule = _require_rule(defaults, ops.SX)
    cz_rule = _require_rule(defaults, ops.CZ)

    m = MatrixImplementationMap()
    m.add(ops.X, x_rule)
    m.add(ops.RZ, rz_rule)
    m.add(ops.SX, sx_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class SCQubitIBMSimulator(_SCQubitSimulator):
    """Simulate an IBM-style superconducting hardware profile.

    Hardware profile:

    - Native gates: ``X``, ``SX``, and ``RZ`` on every qubit; ``CZ`` on
      horizontal or vertical neighbours, in either operand order.
    - Layout: a row-major rectangular grid containing qubits only. A sole
      ``GridRegister`` keeps its coordinates and is placed at the top left.
    - Methods: ``statevector``, ``density_matrix``, ``unitary``, and ``superop``
      are selectable, subject to their usual program restrictions.
    - Noise: ideal unless a model is supplied. ``default_noise_model()``
      creates the optional built-in profile.

    The simulator validates the program as written; it does not decompose,
    route, or schedule operations.
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create an IBM-style constrained simulator.

        Args:
            grid_size: Device shape as ``(rows, columns)``. Both values must
                be positive integers.
            method: ``"statevector"`` (or ``"SV"``), ``"density_matrix"``
                (or ``"DM"``), ``"unitary"``, or ``"superop"``. Names are
                case-insensitive.
            runtime: ``"numba"`` (default, lazy JIT) or ``"numpy"`` (direct
                execution). See ``Simulator`` for runtime-specific
                execution controls.
            noise: Optional ``NoiseModel``. ``None`` keeps the backend ideal;
                pass ``default_noise_model()`` explicitly to use the built-in
                profile.

        Raises:
            TypeError: If ``grid_size`` is not a tuple, or either item is not
                an integer (bools rejected).
            ValueError: If the tuple does not contain exactly two items or
                either item is not positive.
            BackendValidationError: If ``method`` or ``runtime`` is invalid,
                or ``noise`` contains a source this simulator cannot run.
        """
        rows, cols = _validate_grid_size(grid_size)
        super().__init__(
            method=method,
            runtime=runtime,
            implementation_map=fake_superconducting_ibm_implementation_map(rows, cols),
            rows=rows,
            cols=cols,
            noise=noise,
        )

    @classmethod
    def default_noise_model(cls) -> NoiseModel:
        """Return a fresh built-in reference noise model.

        Profile:

        - Coherence: ``T1 = 60 us`` and ``T2 = 48 us``.
        - ``X`` and ``SX``: 20 ns relaxation.
        - ``RZ``: virtual and noiseless.
        - ``CZ``: 50 ns relaxation on each qubit, followed by joint
          depolarizing noise with ``p = 0.001``.
        - Readout: ``P(report 1 | true 0) = 0.02`` and
          ``P(report 0 | true 1) = 0.04``.

        The model is not enabled automatically. Extend the returned
        ``NoiseModel`` if needed, then pass it to ``noise=``.
        """
        noise = NoiseModel()
        damping, dephasing = _THERMAL_RELAXATION.as_channels(_SX_DURATION)
        for gate in (ops.X, ops.SX):
            noise.add(damping, operation=gate)
            noise.add(dephasing, operation=gate)
        cz_damping, cz_dephasing = _THERMAL_RELAXATION.as_channels(_CZ_DURATION)
        for slot in (0, 1):
            noise.add(cz_damping, operation=ops.CZ, target_positions=(slot,))
            noise.add(cz_dephasing, operation=ops.CZ, target_positions=(slot,))
        noise.add(Depolarizing(p=_CZ_DEPOLARIZING_P), operation=ops.CZ)
        noise.add(
            ReadoutConfusion(
                np.array(
                    [
                        [1 - _READOUT_P01, _READOUT_P10],
                        [_READOUT_P01, 1 - _READOUT_P10],
                    ]
                )
            )
        )
        return noise


# --- Google-style backend: RX, RY, RZ, iSwap, CZ ---------------------------


def fake_superconducting_google_implementation_map(
    rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
) -> MatrixImplementationMap:
    """Build the native gate map for a `rows x cols` fake Google-style superconducting backend.

    `RX`, `RY`, and `RZ` are legal on any qubit label (registered uniformly
    via `add`); `iSwap` and `CZ` are legal only on nearest-neighbor grid
    edges, both directions (added with explicit `device_operands`, one call
    per edge, per gate). Every other operation family (including `CX`) has
    no entry and is therefore unsupported.
    """
    rows, cols = _validate_grid_size((rows, cols))
    defaults = default_matrix_implementation_map()
    rx_rule = _require_rule(defaults, ops.RX)
    ry_rule = _require_rule(defaults, ops.RY)
    rz_rule = _require_rule(defaults, ops.RZ)
    iswap_rule = _require_rule(defaults, ops.iSwap)
    cz_rule = _require_rule(defaults, ops.CZ)

    m = MatrixImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.iSwap, iswap_rule, device_operands=edge)
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class SCQubitGoogleSimulator(_SCQubitSimulator):
    """Simulate a Google-style superconducting hardware profile.

    Hardware profile:

    - Native gates: ``RX``, ``RY``, and ``RZ`` on every qubit; ``iSwap`` and
      ``CZ`` on horizontal or vertical neighbours, in either operand order.
    - Layout: a row-major rectangular grid containing qubits only. A sole
      ``GridRegister`` keeps its coordinates and is placed at the top left.
    - Methods: ``statevector``, ``density_matrix``, ``unitary``, and ``superop``
      are selectable, subject to their usual program restrictions.
    - Noise: ideal unless a model is supplied. ``default_noise_model()``
      creates the optional built-in profile.

    The simulator validates the program as written; it does not decompose,
    route, or schedule operations.
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a Google-style constrained simulator.

        Args:
            grid_size: Device shape as ``(rows, columns)``. Both values must
                be positive integers.
            method: ``"statevector"`` (or ``"SV"``), ``"density_matrix"``
                (or ``"DM"``), ``"unitary"``, or ``"superop"``. Names are
                case-insensitive.
            runtime: ``"numba"`` (default, lazy JIT) or ``"numpy"`` (direct
                execution). See ``Simulator`` for runtime-specific
                execution controls.
            noise: Optional ``NoiseModel``. ``None`` keeps the backend ideal;
                pass ``default_noise_model()`` explicitly to use the built-in
                profile.

        Raises:
            TypeError: If ``grid_size`` is not a tuple, or either item is not
                an integer (bools rejected).
            ValueError: If the tuple does not contain exactly two items or
                either item is not positive.
            BackendValidationError: If ``method`` or ``runtime`` is invalid,
                or ``noise`` contains a source this simulator cannot run.
        """
        rows, cols = _validate_grid_size(grid_size)
        super().__init__(
            implementation_map=fake_superconducting_google_implementation_map(
                rows, cols
            ),
            rows=rows,
            cols=cols,
            method=method,
            runtime=runtime,
            noise=noise,
        )

    @classmethod
    def default_noise_model(cls) -> NoiseModel:
        """Return a fresh built-in reference noise model.

        Profile:

        - Coherence: ``T1 = 60 us`` and ``T2 = 48 us``.
        - ``RX``, ``RY``, and ``RZ``: 20 ns relaxation.
        - ``iSwap``: 30 ns relaxation on each qubit, followed by joint
          depolarizing noise with ``p = 0.001``.
        - ``CZ``: 50 ns relaxation on each qubit, followed by joint
          depolarizing noise with ``p = 0.001``.
        - Readout: ``P(report 1 | true 0) = 0.02`` and
          ``P(report 0 | true 1) = 0.04``.

        The model is not enabled automatically. Extend the returned
        ``NoiseModel`` if needed, then pass it to ``noise=``.
        """
        noise = NoiseModel()
        damping, dephasing = _THERMAL_RELAXATION.as_channels(_ROTATION_DURATION)
        for gate in (ops.RX, ops.RY, ops.RZ):
            noise.add(damping, operation=gate)
            noise.add(dephasing, operation=gate)
        iswap_damping, iswap_dephasing = _THERMAL_RELAXATION.as_channels(
            _ISWAP_DURATION
        )
        for slot in (0, 1):
            noise.add(iswap_damping, operation=ops.iSwap, target_positions=(slot,))
            noise.add(iswap_dephasing, operation=ops.iSwap, target_positions=(slot,))
        noise.add(Depolarizing(p=_ISWAP_DEPOLARIZING_P), operation=ops.iSwap)
        cz_damping, cz_dephasing = _THERMAL_RELAXATION.as_channels(_CZ_DURATION)
        for slot in (0, 1):
            noise.add(cz_damping, operation=ops.CZ, target_positions=(slot,))
            noise.add(cz_dephasing, operation=ops.CZ, target_positions=(slot,))
        noise.add(Depolarizing(p=_CZ_DEPOLARIZING_P), operation=ops.CZ)
        noise.add(
            ReadoutConfusion(
                np.array(
                    [
                        [1 - _READOUT_P01, _READOUT_P10],
                        [_READOUT_P01, 1 - _READOUT_P10],
                    ]
                )
            )
        )
        return noise
