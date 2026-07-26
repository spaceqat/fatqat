"""Fake configurable-shape superconducting-grid backends for compiler prototyping.

The default 4x4 device is row-major numbered:

.. code-block:: text

    0   1   2   3
    4   5   6   7
    8   9  10  11
    12 13  14  15

Two concrete backends share a configurable grid, its GridRegister-aware
resource mapping, and nearest-neighbor connectivity (via the private
`_SCQubitSimulator`), differing only in native gate set and
calibration:

- `SCQubitIBMSimulator` - `X`, `SX`, `RZ` (single-qubit, any device
  labels), and `CZ` (nearest-neighbor edges only, both directions stored).
- `SCQubitGoogleSimulator` - `RX`, `RY`, `RZ` (single-qubit, any device
  labels), and `iSwap`/`CZ` (nearest-neighbor edges only, both directions
  stored, both two-qubit gates native at once).

Neither is a realistic device model: no routing, no timing, and ideal by
default unless a noise model is supplied. Each ships a calibration-derived
`default_noise_model()` on demand - the Qiskit ``NoiseModel.from_backend``
workflow - see each class's own docstring for its gate set and noise
profile.

The native-gate-set restriction applies to unitary operations only.
Measurement and reset are resolved by `SimulatorBackend._lower` before any
implementation-map lookup happens (see the `isinstance` dispatch there), so
both backends accept them on any valid device qubit regardless of the
implementation map's contents.
"""

from __future__ import annotations

import numpy as np

from .. import operations as ops
from ..errors import BackendValidationError
from ..implementation import (
    ImplementationMap,
    MatrixImplementation,
    default_matrix_implementation_map,
)
from ..noise import Depolarizing, NoiseModel, relaxation_channels
from ..operations import Operation
from ..program import Program
from ..registers import GridRegister, RegisterRef
from ..resource_layout import ResourceLayout
from .backend_utils import _validate_grid_size
from .simulator_backend import SimulatorBackend

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
    implementation_map: ImplementationMap,
    op: Operation,
) -> MatrixImplementation:
    rule = implementation_map.implementation_for(op)
    if rule is None:
        raise RuntimeError(f"default matrix implementation missing for {op!r}")
    return rule


class _SCQubitSimulator(SimulatorBackend):
    """Shared shape and resource-mapping logic for fake superconducting backends.

    Not part of the public API. `SCQubitIBMSimulator` and
    `SCQubitGoogleSimulator` both subclass this for their configurable device
    shape, GridRegister-aware resource mapping, and the
    `implementation_map` introspection property; each supplies its own
    native-gate implementation map and `default_noise_model`.
    """

    def __init__(
        self,
        implementation_map: ImplementationMap,
        *,
        rows: int,
        cols: int,
        method: str = "statevector",
        runtime: str = "numpy",
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
    def implementation_map(self) -> ImplementationMap:
        """Return a copy of the compiler-facing device-aware implementation map.

        A compiler targeting this device introspects the map rather than
        hardcoding the native gate set: `supported_operations()` lists which
        operation families have any implementation, and
        `device_operands_for(op)` lists the legal device-operand tuples for
        an operation constrained to specific qubits (empty for an operation
        registered uniformly, meaning "legal on any target of the right
        arity").
        """
        return self._impl_map.copy()

    def _resolve_resource_layout(self, program: Program) -> ResourceLayout:
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
        n_subsystems = sum(register.size for register in program.qreg)
        capacity = self._rows * self._cols
        if n_subsystems > capacity:
            raise BackendValidationError(
                f"{name} supports at most {capacity} qubits on its "
                f"{self._rows}x{self._cols} device, got {n_subsystems}"
            )
        dims = (register.dim for register in program.qreg for _ in range(register.size))
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(f"{name} only supports qubit dimensions")
        grid_registers = [r for r in program.qreg if isinstance(r, GridRegister)]
        if len(grid_registers) > 1:
            raise BackendValidationError(
                f"{name} accepts at most one GridRegister per program, "
                f"got {len(grid_registers)}"
            )
        if not grid_registers:
            return super()._resolve_resource_layout(program)

        grid = grid_registers[0]
        if len(program.qreg) != 1:
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
) -> ImplementationMap:
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

    m = ImplementationMap()
    m.add(ops.X, x_rule)
    m.add(ops.RZ, rz_rule)
    m.add(ops.SX, sx_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class SCQubitIBMSimulator(_SCQubitSimulator):
    """Statevector backend constrained to a fake IBM-style superconducting target.

    A thin statevector-method :py:class:`~fatqat.backends.SimulatorBackend`
    specialization: same execution engine, same
    :py:class:`~fatqat.Result`/:py:class:`~fatqat.Job` semantics. The
    differences are a configurable grid device, a fixed native gate set
    (:py:data:`~fatqat.operations.X`,
    :py:data:`~fatqat.operations.SX`, :py:class:`~fatqat.operations.RZ`, and
    nearest-neighbor :py:data:`~fatqat.operations.CZ`), rejecting programs
    that do not fit that device shape (too many qubits, or any non-qubit-dimension
    register), and grid-aware resource mapping (see
    `_resolve_resource_layout`). Qubits here are always "on" - there is no
    atom-loading concept, unlike :py:class:`~fatqat.backends.AtomGridBackend`.
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
        method: str = "statevector",
        runtime: str = "numpy",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a fake IBM-style superconducting backend.

        Args:
            grid_size: Device shape as ``(rows, columns)``. Both values must
                be positive integers.
            method: State representation, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`.
            noise: Optional :py:class:`~fatqat.NoiseModel`, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`. ``None`` (the
                default) keeps the backend ideal; pass
                ``self.default_noise_model()`` for the device's
                calibration-derived profile.

        Raises:
            TypeError: If ``grid_size`` is not a two-item tuple of integers
                (bools rejected).
            ValueError: If ``grid_size`` does not contain exactly two values
                or either value is not positive.
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
        """Build this device's calibration-derived noise model.

        The from-backend workflow: the *backend* authors the model from its
        own device facts, before any user program (or register) exists.
        ``X`` and ``SX`` each carry thermal relaxation converted from the
        device ``T1``/``T2`` and the gate duration (both are physical
        single-qubit pulses here), readout gets an asymmetric confusion
        matrix, and ``CZ`` carries a joint depolarizing channel plus gate-time
        relaxation on each participating qubit. ``RZ`` is
        virtual (zero duration), so it stays noise-free.

        The returned model is a fresh, ordinary
        :py:class:`~fatqat.NoiseModel`: inspect it, extend it with your own
        channels, and pass it back via ``noise=``.

        Examples:
            >>> import fatqat as fq
            >>> Sim = fq.backends.SCQubitIBMSimulator
            >>> backend = Sim(method="statevector", runtime="numpy",
            ...               noise=Sim.default_noise_model())
            >>> program = fq.Program(1, 1)
            >>> program.add(fq.ops.SX, 0)
            >>> program.add(fq.ops.SX, 0)  # SX SX = X, up to a phase
            >>> program.add_measurement(0, 0)
            >>> counts = backend.run(
            ...     program,
            ...     shots=2000,
            ...     simulation_config={"seed": 1, "parallel_mode": "serial"},
            ... ).result().get_counts()
            >>> counts["1"] > 1800  # mostly 1, but noise leaks some 0s
            True
        """
        noise = NoiseModel()
        damping, dephasing = relaxation_channels(_T1, _T2, _SX_DURATION)
        for gate in (ops.X, ops.SX):
            noise.add_noise(gate, damping)
            noise.add_noise(gate, dephasing)
        cz_damping, cz_dephasing = relaxation_channels(_T1, _T2, _CZ_DURATION)
        for slot in (0, 1):
            noise.add_noise(ops.CZ, cz_damping, slots=(slot,))
            noise.add_noise(ops.CZ, cz_dephasing, slots=(slot,))
        noise.add_noise(ops.CZ, Depolarizing(p=_CZ_DEPOLARIZING_P))
        noise.add_readout_error(
            np.array(
                [
                    [1 - _READOUT_P01, _READOUT_P10],
                    [_READOUT_P01, 1 - _READOUT_P10],
                ]
            )
        )
        return noise


# --- Google-style backend: RX, RY, RZ, iSwap, CZ ---------------------------


def fake_superconducting_google_implementation_map(
    rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
) -> ImplementationMap:
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

    m = ImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.iSwap, iswap_rule, device_operands=edge)
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class SCQubitGoogleSimulator(_SCQubitSimulator):
    """Statevector backend constrained to a fake Google-style superconducting target.

    A thin statevector-method :py:class:`~fatqat.backends.SimulatorBackend`
    specialization: same execution engine, same
    :py:class:`~fatqat.Result`/:py:class:`~fatqat.Job` semantics. The
    differences are a configurable grid device, a fixed native gate set
    (:py:class:`~fatqat.operations.RX`,
    :py:class:`~fatqat.operations.RY`, :py:class:`~fatqat.operations.RZ`, and
    nearest-neighbor :py:data:`~fatqat.operations.iSwap` and
    :py:data:`~fatqat.operations.CZ`), rejecting programs that do not fit
    that device shape (too many qubits, or any non-qubit-dimension
    register), and grid-aware resource mapping (see
    `_resolve_resource_layout`). Qubits here are always "on" - there is no
    atom-loading concept, unlike :py:class:`~fatqat.backends.AtomGridBackend`.
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
        method: str = "statevector",
        runtime: str = "numpy",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a fake Google-style superconducting backend.

        Args:
            grid_size: Device shape as ``(rows, columns)``. Both values must
                be positive integers.
            method: State representation, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`.
            noise: Optional :py:class:`~fatqat.NoiseModel`, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`. ``None`` (the
                default) keeps the backend ideal; pass
                ``self.default_noise_model()`` for the device's
                calibration-derived profile.

        Raises:
            TypeError: If ``grid_size`` is not a two-item tuple of integers
                (bools rejected).
            ValueError: If ``grid_size`` does not contain exactly two values
                or either value is not positive.
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
        """Build this device's calibration-derived noise model.

        The from-backend workflow: the *backend* authors the model from its
        own device facts, before any user program (or register) exists.
        ``RX``, ``RY``, and ``RZ`` each carry thermal relaxation converted
        from the device ``T1``/``T2`` and the gate duration. Readout gets an
        asymmetric confusion matrix, and ``iSwap`` and ``CZ`` each carry
        their own joint depolarizing channel plus gate-time relaxation on
        each participating qubit. Unlike the IBM-style backend, ``RZ`` is a
        physical 20 ns rotation here rather than a virtual noiseless gate.

        The returned model is a fresh, ordinary
        :py:class:`~fatqat.NoiseModel`: inspect it, extend it with your own
        channels, and pass it back via ``noise=``.

        Examples:
            >>> import numpy as np
            >>> import fatqat as fq
            >>> Sim = fq.backends.SCQubitGoogleSimulator
            >>> backend = Sim(method="statevector", runtime="numpy",
            ...               noise=Sim.default_noise_model())
            >>> program = fq.Program(1, 1)
            >>> program.add(fq.ops.RX(np.pi), 0)  # RX(pi) = X, up to a phase
            >>> program.add_measurement(0, 0)
            >>> counts = backend.run(
            ...     program,
            ...     shots=2000,
            ...     simulation_config={"seed": 1, "parallel_mode": "serial"},
            ... ).result().get_counts()
            >>> counts["1"] > 1800  # mostly 1, but noise leaks some 0s
            True
        """
        noise = NoiseModel()
        damping, dephasing = relaxation_channels(_T1, _T2, _ROTATION_DURATION)
        for gate in (ops.RX, ops.RY, ops.RZ):
            noise.add_noise(gate, damping)
            noise.add_noise(gate, dephasing)
        iswap_damping, iswap_dephasing = relaxation_channels(_T1, _T2, _ISWAP_DURATION)
        for slot in (0, 1):
            noise.add_noise(ops.iSwap, iswap_damping, slots=(slot,))
            noise.add_noise(ops.iSwap, iswap_dephasing, slots=(slot,))
        noise.add_noise(ops.iSwap, Depolarizing(p=_ISWAP_DEPOLARIZING_P))
        cz_damping, cz_dephasing = relaxation_channels(_T1, _T2, _CZ_DURATION)
        for slot in (0, 1):
            noise.add_noise(ops.CZ, cz_damping, slots=(slot,))
            noise.add_noise(ops.CZ, cz_dephasing, slots=(slot,))
        noise.add_noise(ops.CZ, Depolarizing(p=_CZ_DEPOLARIZING_P))
        noise.add_readout_error(
            np.array(
                [
                    [1 - _READOUT_P01, _READOUT_P10],
                    [_READOUT_P01, 1 - _READOUT_P10],
                ]
            )
        )
        return noise
