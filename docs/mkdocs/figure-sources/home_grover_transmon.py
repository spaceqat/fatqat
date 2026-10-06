"""Run the compiled Grover Program on a three-level TransmonEmulator."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

from _home_grover_plot import draw_distribution
from home_grover_program import (
    TARGET,
    TARGET_INDEX,
    COMPILER_SEED,
    COUPLINGS,
    build_logical_program,
    logical_axis_order,
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
                },
            ],
        },
    },
}
calibration = fq.emulator.TransmonCalibration(calibration_document)

noise = fq.NoiseModel()
for subsystem in model.subsystem_ids:
    noise.add(
        fq.noise.TransitionRelaxation(
            rate=1 / T1_NANOSECONDS,
            coefficients={(1, 0): 1.0, (2, 1): np.sqrt(2.0)},
        ),
        targets=subsystem,
    )
    noise.add(
        fq.noise.PhaseDamping(rate=1 / T2_NANOSECONDS - 1 / (2 * T1_NANOSECONDS)),
        targets=subsystem,
    )

compiler_backend = fq.simulator.SCQubitSimulator(
    num_qubits=3, couplings=COUPLINGS, runtime="numpy"
)
compiled = fq.compiler.compile_to_sc(
    build_logical_program(), compiler_backend, seed=COMPILER_SEED
)
# Bind the compiler's integer sites to this model's named transmons.
resource_layout = fq.ResourceLayout(
    {
        ref: model.subsystem_ids[compiled.resource_layout.device_label(ref)]
        for ref in compiled.resource_layout.refs
    }
)
implementations = fq.emulator.default_transmon_gate_implementation_map(
    model=model,
    calibration=calibration,
)
# Realize the canonical X/SX gates with the calibrated RX pulse recipe.
rx_pulse = implementations.implementation_for(ops.RX)
for subsystem in model.subsystem_ids:
    for gate, angle in ((ops.X, np.pi), (ops.SX, np.pi / 2)):
        implementations.add(
            gate,
            rx_pulse(ops.RX(angle), device_operands=(subsystem,)),
            device_operands=(subsystem,),
        )
emulator = fq.emulator.TransmonEmulator(
    model,
    method="density_matrix",
    noise=noise,
    gate_implementation_map=implementations,
)
result = emulator.run(
    compiled.program,
    resource_layout=resource_layout,
    shots=0,
    result_config={"counts": False, "final_state": True},
).result()

density_matrix = result.get_density_matrix()
physical = np.clip(np.real(np.diag(density_matrix)), 0.0, None)
physical /= physical.sum()
physical = physical.reshape((3, 3, 3)).transpose(logical_axis_order(compiled, result))
binary = np.zeros(8)
leakage = 0.0
for levels in product(range(3), repeat=3):
    probability = physical[levels]
    outcome = sum(
        (level > 0) << (len(levels) - 1 - factor) for factor, level in enumerate(levels)
    )
    binary[outcome] += probability
    if 2 in levels:
        leakage += probability

probabilities = binary / binary.sum()
assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.64046294, atol=1e-3)
assert np.isclose(leakage, 0.0006491483, atol=5e-7)

print(
    f"TransmonEmulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}; "
    f"leakage = {leakage:.8%}"
)
draw_distribution(TRANSMON_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
