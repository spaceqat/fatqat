"""Run the fused Grover Program on a three-level TransmonEmulator."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq

from _home_grover_plot import draw_distribution
from home_grover_program import (
    TARGET,
    TARGET_INDEX,
    build_native_program,
)

TRANSMON_FIGURE = "grover-transmon.png"
T1_NANOSECONDS = 200_000.0
T2_NANOSECONDS = 200_000.0

model_document = {
    "format": {"id": "sc.transmon_exchange", "version": 1},
    "model": {"id": "grover-three-transmon-line", "revision": "2026-08-30"},
    "system": {
        "subsystem_type": "transmon",
        "subsystems": ["q0", "q1", "q2"],
        "control_edges": [
            {"id": "e01", "subsystems": ["q0", "q1"]},
            {"id": "e12", "subsystems": ["q1", "q2"]},
        ],
    },
    "units": {"frequency": "GHz", "anharmonicity": "GHz"},
    "parameters": {
        "subsystems": {
            "q0": {"frequency": 5.10, "anharmonicity": -0.22},
            "q1": {"frequency": 5.22, "anharmonicity": -0.24},
            "q2": {"frequency": 5.34, "anharmonicity": -0.22},
        }
    },
}
model = fq.emulator.TransmonModel.from_document(model_document)

calibration_document = {
    "format": {"id": "sc.transmon_exchange_fixed_pulse", "version": 1},
    "calibration": {
        "id": "grover-three-transmon-line",
        "revision": "2026-08-30",
    },
    "units": {"time": "ns", "frequency": "GHz", "dimensionless": "1"},
    "recipes": {
        "rx_ry": {"duration": 20.0, "drag_coefficient": 1.0},
        "iswap": {"duration": 40.0},
        "cz": {
            "edges": [
                {
                    "canonical_edge": ["q0", "q1"],
                    "recipe": {
                        "detuned_subsystem": "q0",
                        "duration": 60.0,
                        "ramp_duration": 3.0,
                        "park_detuning_ghz": 0.22,
                        "branch_tolerance_ghz": 1e-12,
                    },
                },
                {
                    "canonical_edge": ["q1", "q2"],
                    "recipe": {
                        "detuned_subsystem": "q1",
                        "duration": 60.0,
                        "ramp_duration": 3.0,
                        "park_detuning_ghz": 0.24,
                        "branch_tolerance_ghz": 1e-12,
                    },
                }
            ],
        },
    },
}
calibration = fq.emulator.TransmonCalibration(calibration_document)

noise = fq.NoiseModel()
for subsystem in model.subsystem_ids:
    noise.add(
        fq.noise.ThermalRelaxation(
            t1=T1_NANOSECONDS,
            t2=T2_NANOSECONDS,
        ),
        targets=subsystem,
    )

program = build_native_program()
emulator = fq.emulator.TransmonEmulator(
    model,
    method="density_matrix",
    noise=noise,
    gate_implementation_map=fq.emulator.default_transmon_gate_implementation_map(
        model=model,
        calibration=calibration,
    ),
)
density_matrix = (
    emulator.run(
        program,
        shots=0,
        result_config={"counts": False, "final_state": True},
    )
    .result()
    .get_density_matrix()
)

physical = np.clip(np.real(np.diag(density_matrix)), 0.0, None)
physical /= physical.sum()
physical = physical.reshape((3, 3, 3), order="F")
binary = np.zeros(8)
leakage = 0.0
for levels in product(range(3), repeat=3):
    probability = physical[levels]
    outcome = sum((level > 0) << bit for bit, level in enumerate(levels))
    binary[outcome] += probability
    if 2 in levels:
        leakage += probability

probabilities = binary / binary.sum()
assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.68591064, atol=5e-6)
assert np.isclose(leakage, 0.0004458410, atol=5e-7)

print(
    f"TransmonEmulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}; "
    f"leakage = {leakage:.8%}"
)
draw_distribution(TRANSMON_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
