"""Compile and run an equivalent Grover program on SCQubitSimulator."""

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops
from fatqat.compiler import compile_qasm_to_sc, to_sc_simulator_program

from _home_grover_plot import draw_distribution
from home_grover_program import FUSED_GATES, TARGET, TARGET_INDEX

PROFILE_FIGURE = "grover-sc-profile.png"
T1_SECONDS = 200e-6
T2_SECONDS = 200e-6
SX_DURATION_SECONDS = 20e-9
CZ_DURATION_SECONDS = 50e-9
EDGE_CZ_DEPOLARIZING_P = {
    (0, 1): 0.003,
    (1, 2): 0.003,
}

def build_sc_qasm():
    """Build equivalent QASM for the canonical SC compiler route."""
    statements = ["OPENQASM 3.0;", "qubit[3] q;"]
    for gate in FUSED_GATES:
        if gate[0] == "CZ":
            statements.append(f"cz q[{gate[1]}], q[{gate[2]}];")
            continue
        name, target, quarter_turns = gate
        statements.append(f"{name.lower()}({quarter_turns} * pi / 4) q[{target}];")
    return "\n".join(statements)


SC_QASM = build_sc_qasm()

COUPLINGS = ((0, 1), (1, 2))
compiler_backend = fq.simulator.SCQubitSimulator(
    num_qubits=3,
    couplings=COUPLINGS,
    runtime="numpy",
)
native = compile_qasm_to_sc(SC_QASM, compiler_backend).output
program, resource_layout = to_sc_simulator_program(native)
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

refs_by_site = {
    resource_layout.device_label(ref): ref for ref in resource_layout.refs
}
# Add explicit depolarizing noise on top of the T1/T2 channels.
for edge, depolarizing_p in EDGE_CZ_DEPOLARIZING_P.items():
    noise.add(
        fq.noise.Depolarizing(p=depolarizing_p),
        operation=ops.CZ,
        targets=tuple(refs_by_site[site] for site in edge),
    )

density_matrix = (
    fq.simulator.SCQubitSimulator(
        num_qubits=3,
        couplings=COUPLINGS,
        method="density_matrix",
        runtime="numpy",
        noise=noise,
    )
    .run(
        program,
        shots=0,
        resource_layout=resource_layout,
        result_config={"counts": False, "final_state": True},
    )
    .result()
    .get_density_matrix()
)
probabilities = np.clip(np.real(np.diag(density_matrix)), 0.0, None)
probabilities /= probabilities.sum()

assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.8615386277, atol=5e-7)

print(f"SCQubitSimulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}")
draw_distribution(PROFILE_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
