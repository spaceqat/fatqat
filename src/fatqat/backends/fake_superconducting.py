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
device model: no routing, no timing, and ideal by default. See
``docs/superpowers/specs/2026-07-09-fatqat-target-aware-implementation-map-and-4x4-fake-superconducting-backend-design.md``.

A calibration-derived noise profile is available on demand:
``FakeSuperconducting4x4Backend.default_noise_model()`` builds a
:py:class:`~fatqat.NoiseModel` from the fake device's ``T1``/``T2`` numbers
(relaxation on ``SX``), a ``CZ`` depolarizing rate, and a readout confusion
matrix - the Qiskit ``NoiseModel.from_backend`` workflow. Pass it back via
``noise=`` to run noisy; the backend stays ideal unless asked.

The native-gate-set restriction applies to unitary operations only.
Measurement and reset are resolved by `SimulatorBackend._lower` before any
implementation-map lookup happens (see the `isinstance` dispatch there), so
this backend accepts them on any of the 16 qubits regardless of the
implementation map's contents.
"""

from __future__ import annotations

from typing import Any

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
from ..resource_layout import ResourceLayout
from .simulator_backend import SimulatorBackend

GRID_ROWS = 4
GRID_COLS = 4
N_QUBITS = GRID_ROWS * GRID_COLS

# --- fake calibration profile (the facts a real device would measure) ---
# Deliberately simple uniform numbers in realistic superconducting ranges;
# T2 stays within its physical bound T2 <= 2*T1, and readout is slightly
# asymmetric (reporting 1 for a true 0 is rarer than the reverse), matching
# the usual superconducting readout skew.
_T1 = 60e-6  # seconds
_T2 = 48e-6
_SX_DURATION = 20e-9  # RZ is virtual (zero duration -> no noise)
_CZ_DEPOLARIZING_P = 0.005
_READOUT_P01 = 0.02  # P(report 1 | true 0)
_READOUT_P10 = 0.04  # P(report 0 | true 1)


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
    implementation_map: ImplementationMap,
    op: Operation,
) -> MatrixImplementation:
    rule = implementation_map.implementation_for(op)
    if rule is None:
        raise RuntimeError(f"default matrix implementation missing for {op!r}")
    return rule


def fake_superconducting_4x4_implementation_map() -> ImplementationMap:
    """Build the native gate map for the fake 4x4 superconducting backend.

    `RZ` and `SX` are legal on any qubit label (registered uniformly via
    `add`); `CZ` is legal only on nearest-neighbor grid edges, both
    directions (added with explicit `device_operands`, one call per edge). Every
    other operation family (including `CX`) has no entry and is therefore
    unsupported.
    """
    defaults = default_matrix_implementation_map()
    rz_rule = _require_rule(defaults, ops.RZ)
    sx_rule = _require_rule(defaults, ops.SX)
    cz_rule = _require_rule(defaults, ops.CZ)

    m = ImplementationMap()
    m.add(ops.RZ, rz_rule)
    m.add(ops.SX, sx_rule)
    for edge in _nearest_neighbor_edges():
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class FakeSuperconducting4x4Backend(SimulatorBackend):
    """Statevector backend constrained to a fake 4x4 superconducting target.

    A thin statevector-method :py:class:`~fatqat.backends.SimulatorBackend` specialization: same execution engine, same
    :py:class:`~fatqat.Result`/:py:class:`~fatqat.Job` semantics. The only differences are a fixed 16-qubit
    device, a fixed native gate set (`RZ`, `SX`, nearest-neighbor `CZ`), and
    rejecting programs that do not fit that device shape (too many qubits,
    or any non-qubit-dimension register).
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a fake 4x4 superconducting backend.

        Args:
            options: Same execution-strategy options as :py:class:`~fatqat.backends.SimulatorBackend`
                (``max_workers``, ``parallel_mode``). The implementation map
                is fixed to `fake_superconducting_4x4_implementation_map()`
                and cannot be overridden.
            noise: Optional :py:class:`~fatqat.NoiseModel`, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`. ``None`` (the
                default) keeps the backend ideal; pass
                ``self.default_noise_model()`` for the device's
                calibration-derived profile.
        """
        super().__init__(
            method="statevector",
            options=options,
            implementation_map=fake_superconducting_4x4_implementation_map(),
            noise=noise,
        )

    @classmethod
    def default_noise_model(cls) -> NoiseModel:
        """Build this device's calibration-derived noise model.

        The from-backend workflow: the *backend* authors the model from its
        own device facts, before any user program (or register) exists.
        ``SX`` carries thermal relaxation converted from the device
        ``T1``/``T2`` and the gate duration, readout gets an asymmetric
        confusion matrix, and ``CZ`` carries a joint depolarizing channel.
        ``RZ`` is virtual (zero duration), so it stays noise-free.

        The returned model is a fresh, ordinary
        :py:class:`~fatqat.NoiseModel`: inspect it, extend it with your own
        channels, and pass it back via ``noise=``.

        Examples:
            >>> import fatqat as fq
            >>> Fake = fq.backends.FakeSuperconducting4x4Backend
            >>> backend = Fake(
            ...     options={"parallel_mode": "serial"},
            ...     noise=Fake.default_noise_model(),
            ... )
            >>> program = fq.Program(1, 1)
            >>> program.add(fq.ops.SX, 0)
            >>> program.add(fq.ops.SX, 0)  # SX SX = X, up to a phase
            >>> program.add_measurement(0, 0)
            >>> counts = backend.run(program, shots=2000, seed=1).result().get_counts()
            >>> counts["1"] > 1800  # mostly 1, but noise leaks some 0s
            True
        """
        noise = NoiseModel()
        damping, dephasing = relaxation_channels(_T1, _T2, _SX_DURATION)
        noise.add_noise(ops.SX, damping)
        noise.add_noise(ops.SX, dephasing)
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

    @property
    def implementation_map(self) -> ImplementationMap:
        """Return a copy of the compiler-facing device-aware implementation map.

        A compiler targeting this device introspects the map rather than
        hardcoding the native gate set: ``supported_operations()`` lists which
        operation families have any implementation, and
        ``device_operands_for(op)`` lists the legal device-operand tuples for
        an operation constrained to specific qubits (empty for an operation
        registered uniformly, like ``RZ``/``SX``, meaning "legal on any
        target of the right arity").

        Examples:
            >>> import fatqat as fq
            >>> backend = fq.backends.FakeSuperconducting4x4Backend()
            >>> impl_map = backend.implementation_map
            >>> sorted(op.name for op in impl_map.supported_operations())
            ['CZ', 'RZ', 'SX']
            >>> impl_map.supports(fq.ops.CX)
            False
            >>> sorted(impl_map.device_operands_for(fq.ops.CZ))[:4]
            [(0, 1), (0, 4), (1, 0), (1, 2)]
        """
        return self._impl_map.copy()

    def _resolve_resource_layout(self, program: Program) -> ResourceLayout:
        """Reject any shape the fake device can't run, then map identically.

        A program may declare up to 16 qubits; fewer is fine, since device
        labels are assigned in declaration order (see the base class's
        generic identity mapping), so an N-qubit program always maps onto
        physical qubits `0..N-1`, the same rule used for a full 16-qubit
        program.

        Raises:
            BackendValidationError: If the program declares more than 16
                qubits, or any non-qubit-dimension (`dim != 2`) register.
        """
        n_subsystems = sum(register.size for register in program.qreg)
        if n_subsystems > N_QUBITS:
            raise BackendValidationError(
                f"FakeSuperconducting4x4Backend supports at most 16 qubits, "
                f"got {n_subsystems}"
            )
        dims = (register.dim for register in program.qreg for _ in range(register.size))
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(
                "FakeSuperconducting4x4Backend only supports qubit dimensions"
            )
        return super()._resolve_resource_layout(program)
