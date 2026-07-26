"""Private full-qutrit QuTiP binding for superconducting pulse runs.

This module is the sole home for QuTiP values in the pulse backend.  Its
public-to-the-engine methods accept resolved model values and return NumPy
arrays only; no solver object, ``Qobj``, or qutip-qip value crosses the
backend/result boundary.
"""

from __future__ import annotations

from math import pi
from typing import Any

import numpy as np
from qutip import Qobj, basis, ket2dm, mesolve, qeye, tensor
from scipy.interpolate import CubicSpline

from ...errors import BackendValidationError
from .execution import _PlacedPulseRun
from .resolved import PhaseShift, PhaseSwap, SampledControl
from .superconducting import ControlChannelRef, CouplingRef, PhysicsModel

_EPSILON = 1e-12


class SCQutipAdapter:
    """Stateful private density-matrix adapter for one built SC model."""

    def __init__(self, model: PhysicsModel) -> None:
        self._model = model
        self._dimension = model.physical_dimension ** len(model.subsystems)
        self._annihilation = tuple(
            self._local(ordinal, model.annihilation)
            for ordinal in range(len(model.subsystems))
        )
        self._number = tuple(
            self._local(ordinal, model.number)
            for ordinal in range(len(model.subsystems))
        )
        self._drift = self._build_drift()
        dimensions = [model.physical_dimension] * len(model.subsystems)
        self._state = ket2dm(basis(dimensions, [0] * len(dimensions)))
        self._frames = np.zeros(len(model.subsystems), dtype=float)
        self._time_ns = 0.0

    def density_matrix(self) -> np.ndarray:
        """Return the current physical qutrit density matrix as a NumPy copy."""
        return np.array(self._state.full(), dtype=complex, copy=True)

    def evolve(self, run: _PlacedPulseRun) -> None:
        """Bind and evolve one placed continuous run, retaining private state."""
        if run.start_ns < self._time_ns - _EPSILON:
            raise BackendValidationError("placed pulse runs must be time ordered")
        self._solve((), self._time_ns, run.start_ns)
        terms = []
        action_events: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]] = []
        ordered = sorted(
            enumerate(run.blocks), key=lambda item: (item[1].start_ns, item[0])
        )
        for source_index, block in ordered:
            assert block.start_ns is not None
            self._apply_actions_before(action_events, block.start_ns)
            for child in block.children:
                terms.extend(self._bind_child(child, block.start_ns))
            action_events.append(
                (
                    block.start_ns + block.duration_ns,
                    source_index,
                    block.post_actions,
                )
            )
        self._solve(tuple(terms), run.start_ns, run.end_ns)
        for _end, _source, actions in sorted(action_events, key=lambda item: item[:2]):
            self._apply_actions(actions)
        self._time_ns = run.end_ns

    def _apply_actions_before(
        self,
        events: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]],
        start_ns: float,
    ) -> None:
        ready = [event for event in events if event[0] <= start_ns + _EPSILON]
        if not ready:
            return
        for _end, _source, actions in sorted(ready, key=lambda item: item[:2]):
            self._apply_actions(actions)
        events[:] = [event for event in events if event not in ready]

    def _apply_actions(self, actions: tuple[PhaseShift | PhaseSwap, ...]) -> None:
        for action in actions:
            if isinstance(action, PhaseShift):
                self._frames[self._model.bind_frame(action.frame)] += action.angle_rad
            else:
                first = self._model.bind_frame(action.first)
                second = self._model.bind_frame(action.second)
                self._frames[first], self._frames[second] = (
                    self._frames[second],
                    self._frames[first],
                )

    def _local(self, ordinal: int, matrix: np.ndarray) -> Qobj:
        factors = [qeye(self._model.physical_dimension) for _ in self._model.subsystems]
        factors[ordinal] = Qobj(matrix)
        return tensor(factors)

    def _build_drift(self) -> Qobj:
        # All frequencies use a common q0 rotating frame.  This retains the
        # physically relevant relative free evolution and makes GHz exactly
        # ``2*pi`` angular inverse-nanoseconds at the binding boundary.
        reference_ghz = self._model.subsystems[0].frequency_ghz
        drift = 0 * self._number[0]
        for ordinal, subsystem in enumerate(self._model.subsystems):
            number = self._number[ordinal]
            rotating_frequency = subsystem.frequency_ghz - reference_ghz
            drift += (
                2
                * pi
                * (
                    rotating_frequency * number
                    + subsystem.anharmonicity_ghz
                    * (number * (number - qeye(number.dims[0])))
                    / 2
                )
            )
        for coupling in self._model.couplings:
            first, second = (
                self._model.subsystem_ids.index(identifier)
                for identifier in coupling.subsystem_ids
            )
            drift += (
                2
                * pi
                * coupling.residual_exchange_ghz
                * (
                    self._annihilation[first].dag() * self._annihilation[second]
                    + self._annihilation[first] * self._annihilation[second].dag()
                )
            )
        return drift

    def _bind_child(self, child: SampledControl, block_start_ns: float) -> list[Any]:
        start = block_start_ns + child.start_offset_ns
        stop = start + child.duration_ns
        spline = CubicSpline(child.tlist + start, child.coefficients)

        def coefficient(time: float, _args: Any = None) -> complex:
            del _args
            return 0.0 if time < start or time > stop else complex(spline(time))

        if isinstance(child.channel, ControlChannelRef):
            ordinal = self._model.bind_control(child.channel)
            if child.channel.kind == "detuning":
                return [[self._number[ordinal], coefficient]]
            phase = np.exp(1j * self._frames[ordinal])
            x_operator = self._annihilation[ordinal] + self._annihilation[ordinal].dag()
            y_operator = -1j * (
                self._annihilation[ordinal] - self._annihilation[ordinal].dag()
            )

            def x_coefficient(time: float, args: Any = None) -> float:
                return float((phase * coefficient(time, args)).real)

            def y_coefficient(time: float, args: Any = None) -> float:
                return float((phase * coefficient(time, args)).imag)

            return [[x_operator, x_coefficient], [y_operator, y_coefficient]]
        if isinstance(child.channel, CouplingRef):
            coupling = self._model.couplings[self._model.bind_coupling(child.channel)]
            first, second = (
                self._model.subsystem_ids.index(identifier)
                for identifier in coupling.subsystem_ids
            )
            operator = (
                self._annihilation[first].dag() * self._annihilation[second]
                + self._annihilation[first] * self._annihilation[second].dag()
            )
            return [[operator, coefficient]]
        raise BackendValidationError("pulse control has an unknown channel reference")

    def _solve(self, terms: tuple[Any, ...], start_ns: float, end_ns: float) -> None:
        if end_ns < start_ns - _EPSILON:
            raise BackendValidationError("pulse time must not run backwards")
        if end_ns <= start_ns + _EPSILON:
            return
        hamiltonian = [self._drift, *terms]
        result = mesolve(hamiltonian, self._state, (start_ns, end_ns), c_ops=())
        self._state = result.states[-1]
