"""Compile the shared Grover Program and run it on SCQubitSimulator."""

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.compiler import SCTarget, compile_to_sc

from _home_grover_plot import draw_distribution
from home_grover_program import (
    COMPILER_SEED,
    COUPLINGS,
    TARGET,
    TARGET_INDEX,
    build_logical_program,
    logical_axis_order,
)

PROFILE_FIGURE = "grover-sc-profile.png"
T1_SECONDS = 200e-6
T2_SECONDS = 200e-6
SX_DURATION_SECONDS = 20e-9
CZ_DURATION_SECONDS = 50e-9
EDGE_CZ_DEPOLARIZING_P = {
    (0, 1): 0.003,
    (1, 2): 0.003,
}


compiler_target = SCTarget(num_qubits=3, couplings=COUPLINGS)
compiled = compile_to_sc(build_logical_program(), compiler_target, seed=COMPILER_SEED)
noise = fq.NoiseModel()


def coherence_channels(duration):
    """Return finite simulator channels for the configured T1 and T2."""
    amplitude_p = -np.expm1(-duration / T1_SECONDS)
    pure_dephasing_rate = 1 / T2_SECONDS - 1 / (2 * T1_SECONDS)
    phase_p = -np.expm1(-pure_dephasing_rate * duration)
    return (
        fq.noise.AmplitudeDamping(p=amplitude_p),
        fq.noise.PhaseDamping(p=phase_p),
    )


for operation in (ops.X, ops.SX):
    damping, dephasing = coherence_channels(SX_DURATION_SECONDS)
    noise.add(damping, operation=operation)
    noise.add(dephasing, operation=operation)

damping, dephasing = coherence_channels(CZ_DURATION_SECONDS)
for target_position in (0, 1):
    noise.add(damping, operation=ops.CZ, target_positions=(target_position,))
    noise.add(dephasing, operation=ops.CZ, target_positions=(target_position,))

# Noise target tuples are ordered; routing can use either CZ orientation.
for edge, depolarizing_p in EDGE_CZ_DEPOLARIZING_P.items():
    for ordered_edge in (edge, edge[::-1]):
        noise.add(
            fq.noise.Depolarizing(p=depolarizing_p),
            operation=ops.CZ,
            targets=ordered_edge,
        )

result = (
    fq.simulator.SCQubitSimulator(
        num_qubits=3,
        couplings=COUPLINGS,
        method="density_matrix",
        runtime="numpy",
        noise=noise,
    )
    .run(
        compiled,
        shots=0,
        result_config={"counts": False, "final_state": True},
    )
    .result()
)
density_matrix = result.get_density_matrix()
probabilities = np.clip(np.real(np.diag(density_matrix)), 0.0, None)
probabilities = (
    probabilities.reshape((2, 2, 2))
    .transpose(logical_axis_order(compiled, result))
    .ravel()
)
probabilities /= probabilities.sum()

assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.8164862609, atol=5e-7)

print(f"SCQubitSimulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}")
draw_distribution(PROFILE_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
