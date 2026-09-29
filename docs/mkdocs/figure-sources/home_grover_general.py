"""Run and compile the shared Grover Program, checking ideal equivalence."""

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq

from _home_grover_plot import (
    CIRCUIT_FIGURE,
    GENERAL_FIGURE,
    PROGRAM_DRAW_STYLE,
    draw_distribution,
    style_program_figure,
)
from home_grover_program import (
    COMPILER_SEED,
    COUPLINGS,
    TARGET,
    TARGET_INDEX,
    build_logical_program,
    logical_axis_order,
)

program = build_logical_program()
simulator = fq.simulator.Simulator(runtime="numpy")

state = (
    simulator.run(
        program,
        shots=0,
        result_config={"counts": False, "final_state": True},
    )
    .result()
    .get_statevector()
)
probabilities = np.abs(state) ** 2
probabilities /= probabilities.sum()

compiler_backend = fq.simulator.SCQubitSimulator(
    num_qubits=3, couplings=COUPLINGS, runtime="numpy"
)
compiled = fq.compiler.compile_to_sc(program, compiler_backend, seed=COMPILER_SEED)
compiled_result = compiler_backend.run(
    compiled,
    shots=0,
    result_config={"counts": False, "final_state": True},
).result()
compiled_state = (
    compiled_result.get_statevector()
    .reshape((2, 2, 2))
    .transpose(logical_axis_order(compiled, compiled_result))
    .ravel()
)
overlap = np.vdot(state, compiled_state)

assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.9453125, atol=1e-12, rtol=0)
assert np.isclose(abs(overlap), 1.0, atol=1e-12, rtol=0)
assert np.allclose(compiled_state, overlap * state, atol=1e-12, rtol=0)

print(f"General Simulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}")
circuit_figure = plt.figure(CIRCUIT_FIGURE, figsize=(13.0, 3.2), facecolor="white")
circuit_axis = circuit_figure.add_subplot()
program.draw(ax=circuit_axis, **PROGRAM_DRAW_STYLE)
style_program_figure(circuit_figure, circuit_axis)
draw_distribution(GENERAL_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
