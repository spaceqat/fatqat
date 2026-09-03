"""Shared QuTiP solver policy and public runtime metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

from .._waveforms import SampledWaveform
from ._core.engine import PulsePlanStep
from ._core.pulse import PulseBlock

_BASE_SOLVER_OPTIONS = {"nsteps": 100000}


def _qutip_max_step(plan: Iterable[PulsePlanStep]) -> float | None:
    """Return a pulse-grid-derived integration cap for one prepared run.

    QuTiP's adaptive solvers can step over a time-localized coefficient when
    only the enclosing run endpoints are requested. Its documented pulse
    guidance is to keep ``max_step`` below half the thinnest feature. The
    authored waveform grid is the backend-neutral description of those
    features, so preserve that relationship without assuming a model time
    scale.
    """
    finest_interval = min(
        (
            right - left
            for step in plan
            if isinstance(step, PulseBlock)
            for control in step.controls
            if isinstance(control.waveform, SampledWaveform)
            for left, right in pairwise(control.waveform.times)
        ),
        default=None,
    )
    return None if finest_interval is None else finest_interval / 2.0


def _qutip_solver_options(max_step: float | None) -> dict[str, Any]:
    """Return a fresh options mapping with the program's integration cap."""
    options: dict[str, Any] = dict(_BASE_SOLVER_OPTIONS)
    if max_step is not None:
        options["max_step"] = max_step
    return options


def _qutip_runtime_details(
    solvers: Iterable[str],
    solver_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Return invoked solvers and the options FatQAT overrides."""
    names = tuple(sorted(set(solvers)))
    if not names:
        solver: str | tuple[str, ...] = "none"
    elif len(names) == 1:
        solver = names[0]
    else:
        solver = names
    return {
        "solver": solver,
        "solver_options": {} if not names else dict(solver_overrides),
    }


class _QutipRuntime:
    """Own the solver options and invocation facts for one pulse execution."""

    def __init__(self, *, max_step: float | None = None) -> None:
        self._program_max_step = max_step
        self._solver_options = _qutip_solver_options(max_step)
        self._solvers_used: set[str] = set()

    def options_for(self, plan: Iterable[PulsePlanStep]) -> dict[str, Any]:
        """Return program options, deriving a cap for direct adapter use."""
        if self._program_max_step is None:
            self._solver_options = _qutip_solver_options(_qutip_max_step(plan))
        return self._solver_options

    def record_solver(self, solver: str) -> None:
        """Record one QuTiP solver that was invoked."""
        self._solvers_used.add(solver)

    def details(self) -> dict[str, Any]:
        """Return normalized public facts about the completed execution."""
        return _qutip_runtime_details(self._solvers_used, self._solver_options)


__all__: list[str] = []
