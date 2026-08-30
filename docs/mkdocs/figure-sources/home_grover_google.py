"""Run the fused Grover Program on SCQubitGoogleSimulator."""

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

from _home_grover_plot import draw_distribution
from home_grover_program import (
    TARGET,
    TARGET_INDEX,
    build_native_program,
)

PROFILE_FIGURE = "grover-google-profile.png"
T1_SECONDS = 200e-6
T2_SECONDS = 200e-6
ROTATION_DURATION_SECONDS = 20e-9
CZ_DURATION_SECONDS = 50e-9
EDGE_CZ_FIDELITIES = {
    (0, 1): 0.998,
    (1, 2): 0.996,
}

program = build_native_program()
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

for operation in (ops.RX, ops.RY, ops.RZ):
    damping, dephasing = coherence_channels(ROTATION_DURATION_SECONDS)
    noise.add(damping, operation=operation)
    noise.add(dephasing, operation=operation)

damping, dephasing = coherence_channels(CZ_DURATION_SECONDS)
for target_position in (0, 1):
    noise.add(damping, operation=ops.CZ, target_positions=(target_position,))
    noise.add(dephasing, operation=ops.CZ, target_positions=(target_position,))

# Reserve only the error budget left after T1/T2 decay for the joint
# depolarizing channel, so each combined CZ channel reaches its stated
# two-qubit average gate fidelity.
single_qubit_entanglement_fidelity = (
    1
    + np.exp(-CZ_DURATION_SECONDS / T1_SECONDS)
    + 2 * np.exp(-CZ_DURATION_SECONDS / T2_SECONDS)
) / 4
cz_relaxation_fidelity = (4 * single_qubit_entanglement_fidelity**2 + 1) / 5

qubits = program.quantum_registers[0]
for edge, fidelity in EDGE_CZ_FIDELITIES.items():
    depolarizing_p = (cz_relaxation_fidelity - fidelity) / (
        cz_relaxation_fidelity - 1 / 4
    )
    assert 0 <= depolarizing_p <= 1
    assert np.isclose(
        (1 - depolarizing_p) * cz_relaxation_fidelity + depolarizing_p / 4,
        fidelity,
    )
    noise.add(
        fq.noise.Depolarizing(p=depolarizing_p),
        operation=ops.CZ,
        targets=tuple(qubits[index] for index in edge),
    )

density_matrix = (
    fq.simulator.SCQubitGoogleSimulator(
        grid_size=(1, 3),
        method="density_matrix",
        runtime="numpy",
        noise=noise,
    )
    .run(program, shots=0, result_config={"counts": False, "final_state": True})
    .result()
    .get_density_matrix()
)
probabilities = np.clip(np.real(np.diag(density_matrix)), 0.0, None)
probabilities /= probabilities.sum()

assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.8506314517, atol=5e-7)

print(f"SCQubitGoogleSimulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}")
draw_distribution(PROFILE_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
