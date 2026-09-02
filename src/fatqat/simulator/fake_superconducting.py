"""Fake superconducting backends with configurable coupling graphs.

The default test target uses 16 integer-labeled sites and the couplings of
this 4x4 graph:

.. code-block:: text

    0   1   2   3
    4   5   6   7
    8   9  10  11
    12 13  14  15

The grid is only default test data. Both concrete backends accept an arbitrary
``num_qubits`` plus undirected ``couplings`` and otherwise differ only in
native gate set and calibration:

- `SCQubitIBMSimulator` - `X`, `SX`, `RZ` (single-qubit, any device
  labels), and `CZ` (coupled pairs only, both directions stored).
- `SCQubitGoogleSimulator` - `RX`, `RY`, `RZ` (single-qubit, any device
  labels), and `iSwap`/`CZ` (coupled pairs only, both directions
  stored, both two-qubit gates native at once).

Neither is a realistic device model: no routing, no timing, and ideal by
default unless a noise model is supplied. Each ships a calibration-derived
`default_noise_model()` on demand - the Qiskit ``NoiseModel.from_backend``
workflow - see each class's own docstring for its gate set and noise
profile.

The native-gate-set restriction applies to unitary operations only.
Measurement and reset are resolved by `Simulator._lower` before any
implementation-map lookup happens (see the `isinstance` dispatch there), so
both backends accept them on any valid device qubit regardless of the
implementation map's contents.
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
from ..noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ReadoutConfusion,
    ThermalRelaxation,
)
from ..operations import Operation
from ..program import Program
from ..resource_layout import DeviceOperand, ResourceLayout
from .simulator import Simulator

DEFAULT_NUM_QUBITS = 16

# --- fake calibration profile (the facts a real device would measure) ---
# Deliberately simple uniform numbers in realistic superconducting ranges;
# T2 stays within its physical bound T2 <= 2*T1, and readout is slightly
# asymmetric (reporting 1 for a true 0 is rarer than the reverse), matching
# the usual superconducting readout skew.
_T1 = 60e-6  # seconds
_T2 = 48e-6
_THERMAL_RELAXATION = ThermalRelaxation(t1=_T1, t2=_T2)
_AMPLITUDE_SOURCE = AmplitudeDamping(rate=_THERMAL_RELAXATION.amplitude_rate)
_PHASE_SOURCE = PhaseDamping(rate=_THERMAL_RELAXATION.pure_dephasing_rate)
_SX_DURATION = 20e-9  # IBM-style RZ is virtual (zero duration -> no noise)
_ROTATION_DURATION = 20e-9  # Google-style RX/RY/RZ: all physical rotations
_CZ_DURATION = 50e-9
_ISWAP_DURATION = 30e-9
_CZ_DEPOLARIZING_P = 0.001
_ISWAP_DEPOLARIZING_P = 0.001
_READOUT_P01 = 0.02  # P(report 1 | true 0)
_READOUT_P10 = 0.04  # P(report 0 | true 1)


def _calibrated_relaxation_channels(
    duration: float,
) -> tuple[AmplitudeDamping, PhaseDamping]:
    """Return finite qubit channels for the fake T1/T2 calibration."""
    return (
        AmplitudeDamping(p=_AMPLITUDE_SOURCE.as_probability(duration)),
        PhaseDamping(p=_PHASE_SOURCE.as_probability(duration)),
    )


def _grid_couplings(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """Build undirected row-major grid edges used by the default test target."""
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            q = row * cols + col
            if col + 1 < cols:
                edges.append((q, q + 1))
            if row + 1 < rows:
                edges.append((q, q + cols))
    return tuple(edges)


DEFAULT_COUPLINGS = _grid_couplings(4, 4)


def _validate_num_qubits(num_qubits: int) -> int:
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer")
    return num_qubits


def _normalize_couplings(
    num_qubits: int, couplings: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge in couplings:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError("couplings must contain two-integer tuples")
        first, second = edge
        if type(first) is not int or type(second) is not int:
            raise TypeError("coupling endpoints must be integers")
        if not 0 <= first < num_qubits or not 0 <= second < num_qubits:
            raise ValueError("coupling endpoint is outside device_sites")
        if first == second:
            raise ValueError("coupling endpoints must be distinct")
        canonical = (min(first, second), max(first, second))
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return tuple(normalized)


def _directed_couplings(
    couplings: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        directed
        for first, second in couplings
        for directed in ((first, second), (second, first))
    )


def _require_rule(
    implementation_map: MatrixImplementationMap,
    op: Operation,
) -> MatrixImplementation:
    rule = implementation_map.implementation_for(op)
    if rule is None:
        raise RuntimeError(f"default matrix implementation missing for {op!r}")
    return rule


class _SCQubitSimulator(Simulator):
    """Shared capacity and resource mapping for fake superconducting backends.

    Not part of the public API. `SCQubitIBMSimulator` and
    `SCQubitGoogleSimulator` both subclass this for capacity, declaration-order
    resource mapping, and implementation-map introspection; each supplies its
    own native-gate implementation map and `default_noise_model`.
    """

    def __init__(
        self,
        implementation_map: MatrixImplementationMap,
        *,
        num_qubits: int,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        self._num_qubits = num_qubits
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
        return frozenset(range(self._num_qubits))

    def _physical_dimension(
        self, device_operand: DeviceOperand, resource_layout: ResourceLayout
    ) -> int:
        return 2

    def _default_resource_layout(self, program: Program) -> ResourceLayout:
        """Validate capacity/dimensions, then bind in declaration order.

        Raises:
            BackendValidationError: If the program declares more than this
                backend's capacity or any non-qubit-dimension (`dim != 2`)
                register.
        """
        name = type(self).__name__
        n_subsystems = sum(register.size for register in program.quantum_registers)
        capacity = self._num_qubits
        if n_subsystems > capacity:
            raise BackendValidationError(
                f"{name} supports at most {capacity} qubits, got {n_subsystems}"
            )
        dims = (
            register.dim
            for register in program.quantum_registers
            for _ in range(register.size)
        )
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(f"{name} only supports qubit dimensions")
        return super()._default_resource_layout(program)


# --- IBM-style backend: X, SX, RZ, CZ --------------------------------------


def fake_superconducting_ibm_implementation_map(
    couplings: tuple[tuple[int, int], ...] = DEFAULT_COUPLINGS,
) -> MatrixImplementationMap:
    """Build the native gate map for an IBM-style coupling graph.

    `X`, `SX`, and `RZ` are legal on any qubit label (registered uniformly
    via `add`); `CZ` is legal only on supplied coupling edges, both
    directions (added with explicit `device_operands`, one call per edge).
    Every other operation family (including `CX`) has no entry and is
    therefore unsupported.
    """
    defaults = default_matrix_implementation_map()
    x_rule = _require_rule(defaults, ops.X)
    rz_rule = _require_rule(defaults, ops.RZ)
    sx_rule = _require_rule(defaults, ops.SX)
    cz_rule = _require_rule(defaults, ops.CZ)

    m = MatrixImplementationMap()
    m.add(ops.X, x_rule)
    m.add(ops.RZ, rz_rule)
    m.add(ops.SX, sx_rule)
    for edge in _directed_couplings(couplings):
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class SCQubitIBMSimulator(_SCQubitSimulator):
    """Simulate an IBM-style superconducting hardware profile.

    A thin statevector-method :py:class:`~fatqat.simulator.Simulator`
    specialization: same execution engine, same
    :py:class:`~fatqat.Result`/:py:class:`~fatqat.Job` semantics. The
    differences are a configurable coupling graph, a fixed native gate set
    (:py:data:`~fatqat.operations.X`,
    :py:data:`~fatqat.operations.SX`, :py:class:`~fatqat.operations.RZ`, and
    coupled :py:data:`~fatqat.operations.CZ`), rejecting programs
    with too many qubits or any non-qubit-dimension register, and
    declaration-order resource mapping (see
    `_resolve_resource_layout`). Qubits here are always "on" - there is no
    atom-loading concept, unlike :py:class:`~fatqat.simulator.AtomArraySimulator`.
    The simulator validates the program as written; it does not decompose,
    route, or schedule operations.
    """

    def __init__(
        self,
        *,
        num_qubits: int = DEFAULT_NUM_QUBITS,
        couplings: tuple[tuple[int, int], ...] = DEFAULT_COUPLINGS,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create an IBM-style constrained simulator.

        Args:
            num_qubits: Number of integer-labeled device sites.
            couplings: Undirected pairs of connected device sites.
            method: State representation, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            noise: Optional :py:class:`~fatqat.NoiseModel`, exactly as on
                :py:class:`~fatqat.simulator.Simulator`. ``None`` (the
                default) keeps the backend ideal; pass
                ``self.default_noise_model()`` for the device's
                calibration-derived profile.

        Raises:
            TypeError: If the site count or coupling endpoints are not integers.
            ValueError: If the site count or coupling endpoints are invalid.
        """
        num_qubits = _validate_num_qubits(num_qubits)
        couplings = _normalize_couplings(num_qubits, couplings)
        super().__init__(
            method=method,
            runtime=runtime,
            implementation_map=fake_superconducting_ibm_implementation_map(couplings),
            num_qubits=num_qubits,
            noise=noise,
        )

    @property
    def device_sites(self) -> tuple[int, ...]:
        """Return every integer-labeled physical qubit on this target."""
        return tuple(range(self._num_qubits))

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
        damping, dephasing = _calibrated_relaxation_channels(_SX_DURATION)
        for gate in (ops.X, ops.SX):
            noise.add(damping, operation=gate)
            noise.add(dephasing, operation=gate)
        cz_damping, cz_dephasing = _calibrated_relaxation_channels(_CZ_DURATION)
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
    couplings: tuple[tuple[int, int], ...] = DEFAULT_COUPLINGS,
) -> MatrixImplementationMap:
    """Build the native gate map for a Google-style coupling graph.

    `RX`, `RY`, and `RZ` are legal on any qubit label (registered uniformly
    via `add`); `iSwap` and `CZ` are legal only on supplied coupling
    edges, both directions (added with explicit `device_operands`, one call
    per edge, per gate). Every other operation family (including `CX`) has
    no entry and is therefore unsupported.
    """
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
    for edge in _directed_couplings(couplings):
        m.add(ops.iSwap, iswap_rule, device_operands=edge)
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class SCQubitGoogleSimulator(_SCQubitSimulator):
    """Simulate a Google-style superconducting hardware profile.

    A thin statevector-method :py:class:`~fatqat.simulator.Simulator`
    specialization: same execution engine, same
    :py:class:`~fatqat.Result`/:py:class:`~fatqat.Job` semantics. The
    differences are a configurable coupling graph, a fixed native gate set
    (:py:class:`~fatqat.operations.RX`,
    :py:class:`~fatqat.operations.RY`, :py:class:`~fatqat.operations.RZ`, and
    coupled :py:data:`~fatqat.operations.iSwap` and
    :py:data:`~fatqat.operations.CZ`), rejecting programs with too many
    qubits or any non-qubit-dimension register, and declaration-order resource
    mapping (see `_resolve_resource_layout`). Qubits here are always "on" - there is no
    atom-loading concept, unlike :py:class:`~fatqat.simulator.AtomArraySimulator`.
    The simulator validates the program as written; it does not decompose,
    route, or schedule operations.
    """

    def __init__(
        self,
        *,
        num_qubits: int = DEFAULT_NUM_QUBITS,
        couplings: tuple[tuple[int, int], ...] = DEFAULT_COUPLINGS,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a Google-style constrained simulator.

        Args:
            num_qubits: Number of integer-labeled device sites.
            couplings: Undirected pairs of connected device sites.
            method: State representation, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            noise: Optional :py:class:`~fatqat.NoiseModel`, exactly as on
                :py:class:`~fatqat.simulator.Simulator`. ``None`` (the
                default) keeps the backend ideal; pass
                ``self.default_noise_model()`` for the device's
                calibration-derived profile.

        Raises:
            TypeError: If the site count or coupling endpoints are not integers.
            ValueError: If the site count or coupling endpoints are invalid.
        """
        num_qubits = _validate_num_qubits(num_qubits)
        couplings = _normalize_couplings(num_qubits, couplings)
        super().__init__(
            implementation_map=fake_superconducting_google_implementation_map(
                couplings
            ),
            num_qubits=num_qubits,
            method=method,
            runtime=runtime,
            noise=noise,
        )

    @property
    def device_sites(self) -> tuple[int, ...]:
        """Return every integer-labeled physical qubit on this target."""
        return tuple(range(self._num_qubits))

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
        damping, dephasing = _calibrated_relaxation_channels(_ROTATION_DURATION)
        for gate in (ops.RX, ops.RY, ops.RZ):
            noise.add(damping, operation=gate)
            noise.add(dephasing, operation=gate)
        iswap_damping, iswap_dephasing = _calibrated_relaxation_channels(
            _ISWAP_DURATION
        )
        for slot in (0, 1):
            noise.add(iswap_damping, operation=ops.iSwap, target_positions=(slot,))
            noise.add(iswap_dephasing, operation=ops.iSwap, target_positions=(slot,))
        noise.add(Depolarizing(p=_ISWAP_DEPOLARIZING_P), operation=ops.iSwap)
        cz_damping, cz_dephasing = _calibrated_relaxation_channels(_CZ_DURATION)
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
