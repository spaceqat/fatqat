"""Individual LQCloud QZ01 calibration values exported on 2026-09-12.

Times are seconds. T2E drives idle relaxation; T2* is retained for provenance.
Reported XEB/CZ fidelities are interpreted as average gate fidelities when
constructing effective depolarizing channels. These are fixed snapshot data,
not live calibration values.
"""

from __future__ import annotations

from dataclasses import dataclass


def _depolarizing_p_from_average_gate_fidelity(
    fidelity: float, dimension: int
) -> float:
    """Convert average gate fidelity to fatqat's depolarizing parameter.

    Fatqat defines the channel as ``E(rho) = (1-p) rho + p I/d``. Its average
    fidelity relative to the ideal gate is ``1 - p * (d - 1) / d``, hence
    ``p = d * (1 - fidelity) / (d - 1)``.
    """
    if not 0.0 <= fidelity <= 1.0:
        raise ValueError(f"fidelity must be in [0, 1], got {fidelity!r}")
    if dimension < 2:
        raise ValueError(f"dimension must be at least 2, got {dimension!r}")
    return dimension * (1.0 - fidelity) / (dimension - 1.0)


@dataclass(frozen=True)
class _QEC17QubitCalibration:
    qubit: int
    t1: float
    t2_echo: float
    t2_star: float
    xeb_fidelity: float
    f00: float
    f11: float

    @property
    def rotation_depolarizing_p(self) -> float:
        return _depolarizing_p_from_average_gate_fidelity(
            self.xeb_fidelity,
            dimension=2,
        )

    @property
    def readout_p01(self) -> float:
        return 1.0 - self.f00

    @property
    def readout_p10(self) -> float:
        return 1.0 - self.f11


@dataclass(frozen=True)
class _QEC17CouplerCalibration:
    edge: tuple[int, int]
    cz_fidelity: float

    @property
    def cz_depolarizing_p(self) -> float:
        return _depolarizing_p_from_average_gate_fidelity(self.cz_fidelity, dimension=4)


# Ordered by physical qubit id, so tuple index == qubit id.
_QEC17_QUBIT_CALIBRATIONS: tuple[_QEC17QubitCalibration, ...] = (
    _QEC17QubitCalibration(0, 38.16e-6, 7.85e-6, 3.75e-6, 0.99948, 0.994, 0.985),
    _QEC17QubitCalibration(1, 39.73e-6, 7.21e-6, 3.11e-6, 0.99935, 0.995, 0.980),
    _QEC17QubitCalibration(2, 32.05e-6, 12.94e-6, 2.50e-6, 0.99919, 0.996, 0.975),
    _QEC17QubitCalibration(3, 43.12e-6, 8.76e-6, 2.41e-6, 0.99935, 0.997, 0.979),
    _QEC17QubitCalibration(4, 48.90e-6, 8.39e-6, 2.96e-6, 0.99900, 0.991, 0.978),
    _QEC17QubitCalibration(5, 42.74e-6, 9.67e-6, 2.19e-6, 0.99938, 0.998, 0.984),
    _QEC17QubitCalibration(6, 56.16e-6, 8.00e-6, 3.11e-6, 0.99926, 0.996, 0.983),
    _QEC17QubitCalibration(7, 51.08e-6, 7.60e-6, 1.59e-6, 0.99916, 0.995, 0.977),
    _QEC17QubitCalibration(8, 44.80e-6, 10.66e-6, 2.23e-6, 0.99937, 0.996, 0.978),
    _QEC17QubitCalibration(9, 46.24e-6, 10.10e-6, 2.84e-6, 0.99922, 0.994, 0.982),
    _QEC17QubitCalibration(10, 34.88e-6, 14.16e-6, 4.86e-6, 0.99912, 0.995, 0.967),
    _QEC17QubitCalibration(11, 27.97e-6, 14.95e-6, 5.99e-6, 0.99899, 0.993, 0.982),
    _QEC17QubitCalibration(12, 29.37e-6, 7.40e-6, 4.03e-6, 0.99935, 0.998, 0.976),
    _QEC17QubitCalibration(13, 33.77e-6, 9.16e-6, 2.06e-6, 0.99941, 0.999, 0.984),
    _QEC17QubitCalibration(14, 30.16e-6, 12.45e-6, 3.77e-6, 0.99922, 0.999, 0.981),
    _QEC17QubitCalibration(15, 30.00e-6, 10.28e-6, 3.03e-6, 0.99952, 0.999, 0.987),
    _QEC17QubitCalibration(16, 23.28e-6, 8.62e-6, 3.47e-6, 0.99914, 0.993, 0.973),
)


# Edges are canonicalized as (data qubit, syndrome qubit).
_QEC17_COUPLER_CALIBRATIONS: tuple[_QEC17CouplerCalibration, ...] = (
    _QEC17CouplerCalibration((0, 9), 0.99601),
    _QEC17CouplerCalibration((1, 9), 0.99501),
    _QEC17CouplerCalibration((1, 10), 0.99176),
    _QEC17CouplerCalibration((1, 13), 0.99579),
    _QEC17CouplerCalibration((2, 10), 0.99400),
    _QEC17CouplerCalibration((2, 13), 0.99259),
    _QEC17CouplerCalibration((3, 9), 0.99593),
    _QEC17CouplerCalibration((3, 11), 0.99552),
    _QEC17CouplerCalibration((4, 9), 0.99520),
    _QEC17CouplerCalibration((4, 10), 0.99497),
    _QEC17CouplerCalibration((4, 11), 0.99432),
    _QEC17CouplerCalibration((4, 12), 0.99420),
    _QEC17CouplerCalibration((5, 10), 0.99568),
    _QEC17CouplerCalibration((5, 12), 0.99442),
    _QEC17CouplerCalibration((5, 14), 0.99668),
    _QEC17CouplerCalibration((6, 11), 0.99492),
    _QEC17CouplerCalibration((6, 15), 0.99615),
    _QEC17CouplerCalibration((7, 11), 0.99384),
    _QEC17CouplerCalibration((7, 12), 0.99130),
    _QEC17CouplerCalibration((7, 15), 0.99349),
    _QEC17CouplerCalibration((8, 12), 0.99525),
    _QEC17CouplerCalibration((8, 14), 0.99410),
    _QEC17CouplerCalibration((3, 16), 0.99400),
    _QEC17CouplerCalibration((0, 16), 0.99513),
)
