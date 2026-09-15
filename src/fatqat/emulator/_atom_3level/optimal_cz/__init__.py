"""Time-optimal phase control for the two-atom Rydberg CZ gate.

This package implements the optimization method and the error analysis of

    S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates for
    Rydberg Atoms", Quantum 6, 712 (2022), doi:10.22331/q-2022-06-20-712,
    arXiv:2202.00903,

for the two-atom controlled-Z gate driven by a single global laser in the
Rydberg blockade regime.  It is the "optimal CZ" companion of the private
three-level atom emulator: the optimizer works in the dimensionless units of
the paper, and :mod:``.pulses`` converts a result into the physical pulse
values that :class:``~..backend.Atom3LevelEmulator`` executes.

How the optimum is found
------------------------
:func:``time_optimal_cz_pulse`` follows the paper exactly, in three stages.

1. **Time-optimal GRAPE** (:mod:``.grape``, paper Secs. 2.2.1 and 3.1).  The
   laser amplitude is pinned at ``Omega_max`` and the phase is piecewise
   constant on ``N`` intervals.  Gradients of the averaged gate error, paper
   Eq. (7), are computed analytically for all ``N + 1`` parameters, including
   the free single-qubit phase ``theta``, and a limited-memory BFGS method
   minimizes them.  Sweeping the duration downwards with warm starts gives the
   threshold duration ``T*`` through the fit ``1 - F = A (T* - T)^2`` of paper
   Eq. (17).
2. **PMP reconstruction** (:mod:``.pmp``, paper Sec. 4).  The optimal control
   of a time-optimal trajectory is fixed by the initial costates through paper
   Eq. (21).  The costates are estimated from the GRAPE pulse with the
   singular-value construction of paper Eqs. (24)-(26) and refined with
   Powell's method, which turns the staircase into a smooth pulse described by
   a handful of numbers.
3. **Verification and error analysis** (:mod:``.error_analysis``, paper
   Secs. 5-7).  The exported pulse is run through the emulator at a finite
   ``C6 / d^6`` blockade and with a Lindblad Rydberg decay, and the result is
   compared with the estimates ``1 - F = Gamma T_R`` and
   ``1 - F = alpha Omega_max^2 / (B^2 (T Omega_max)^2)`` of paper Eq. (35).

Example
-------
Find the time-optimal pulse, then export it to the emulator::

    from fatqat.emulator._atom_3level import Atom3LevelModel
    from fatqat.emulator._atom_3level.optimal_cz import (
        time_optimal_cz_pulse,
        cz_gate_implementation_map,
    )

    pulse, fit = time_optimal_cz_pulse(pieces=99, seed=0)
    print(fit.optimal_duration, pulse.gate_error(), pulse.rydberg_time())

    model = Atom3LevelModel.from_document(document)
    backend = Atom3LevelEmulator(
        model,
        arrangement=AtomArrangement.chain(2, 7.0),
        method="unitary",
        gate_implementation_map=cz_gate_implementation_map(
            model, pulse, omega_max=2 * 3.14159 * 5.0
        ),
    )

Every function is documented with the equation of the paper that it
implements, and the module docstrings repeat the reference so that a reader of
a single module still sees the source.
"""

from __future__ import annotations

from typing import Any

from .grape import (
    CONVERGED_GATE_ERROR,
    TimeOptimalFit,
    TimeOptimalScan,
    fit_time_optimal_duration,
    optimize_cz_pulse,
    scan_durations,
    time_optimal_cz_pulse,
)
from .pmp import (
    PAPER_CZ_COSTATES,
    PAPER_CZ_COSTATE_DURATION,
    PMPCostates,
    PMPReconstruction,
    costates_from_paper_table,
    extract_costates,
    pmp_cz_pulse,
    reconstruct_pulse,
    refine_costates,
)
from .problem import (
    BLOCK_COUPLINGS,
    CZPulse,
    averaged_gate_fidelity,
    block_amplitudes,
    excited_state_populations,
    phase_gate_phases,
    rydberg_time,
)
from .pulses import (
    DEFAULT_SAMPLE_COUNT,
    cz_gate_implementation_map,
    cz_pulse_definition,
    cz_waveform,
)

__all__ = [
    "BLOCK_COUPLINGS",
    "CONVERGED_GATE_ERROR",
    "CZPulse",
    "DEFAULT_SAMPLE_COUNT",
    "PAPER_CZ_COSTATES",
    "PAPER_CZ_COSTATE_DURATION",
    "PMPCostates",
    "PMPReconstruction",
    "TimeOptimalFit",
    "TimeOptimalScan",
    "averaged_gate_fidelity",
    "block_amplitudes",
    "costates_from_paper_table",
    "cz_gate_implementation_map",
    "cz_pulse_definition",
    "cz_waveform",
    "excited_state_populations",
    "extract_costates",
    "fit_time_optimal_duration",
    "optimize_cz_pulse",
    "phase_gate_phases",
    "pmp_cz_pulse",
    "reconstruct_pulse",
    "refine_costates",
    "rydberg_time",
    "scan_durations",
    "time_optimal_cz_pulse",
]

#: Names that live in :mod:``.error_analysis`` and pull in the emulator stack.
#: They are imported on first access so that the optimizer alone only needs
#: NumPy and SciPy.
_ERROR_ANALYSIS_NAMES = frozenset(
    {
        "COMPUTATIONAL_INDICES",
        "CZGateMetrics",
        "DecayFidelity",
        "GateErrorBudget",
        "emulated_cz_metrics",
        "emulated_decay_fidelity",
        "finite_blockade_error",
        "gate_error_budget",
        "optimal_rabi_frequency",
        "quadratic_blockade_coefficient",
        "rydberg_decay_error",
    }
)


def __getattr__(name: str) -> Any:
    """Import the emulator-backed error analysis on first use."""
    if name in _ERROR_ANALYSIS_NAMES:
        from . import error_analysis

        return getattr(error_analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the module attributes, including the lazy error analysis."""
    return sorted(__all__ + sorted(_ERROR_ANALYSIS_NAMES))
