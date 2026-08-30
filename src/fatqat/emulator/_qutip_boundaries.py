"""Private representation-neutral QuTiP boundary algebra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qutip import Qobj


def _sample_projective_qutip_state(
    state: Qobj,
    projectors: Sequence[Qobj],
    rng: np.random.Generator,
) -> tuple[int, Qobj]:
    """Sample projectors and return the physical outcome and posterior state."""
    if state.isoper:
        branches = tuple(projector * state * projector for projector in projectors)
        weights = np.asarray(
            [float(np.real(branch.tr())) for branch in branches],
            dtype=float,
        )
    elif state.isket:
        branches = tuple(projector * state for projector in projectors)
        weights = np.asarray([branch.norm() ** 2 for branch in branches], dtype=float)
    else:
        raise TypeError("projective measurement requires a ket or density operator")
    if np.any(weights < -1e-10):
        raise RuntimeError("physical measurement produced invalid probabilities")
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("physical measurement produced invalid probabilities")
    outcome = int(rng.choice(len(branches), p=weights / total))
    normalization = weights[outcome] if state.isoper else np.sqrt(weights[outcome])
    return outcome, branches[outcome] / normalization


def _apply_qutip_reset(
    state: Qobj,
    operators: Sequence[Qobj],
    rng: np.random.Generator,
) -> Qobj:
    """Apply an exact reset channel or sample one pure-state Kraus branch."""
    if state.isoper:
        return sum(operator * state * operator.dag() for operator in operators)
    if not state.isket:
        raise TypeError("reset requires a ket or density operator")
    branches = tuple(operator * state for operator in operators)
    weights = np.asarray([branch.norm() ** 2 for branch in branches], dtype=float)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("physical reset produced invalid branch probabilities")
    branch = int(rng.choice(len(branches), p=weights / total))
    return branches[branch] / np.sqrt(weights[branch])


__all__: list[str] = []
