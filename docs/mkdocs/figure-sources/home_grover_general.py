"""Run the fused Grover Program on the general Simulator."""

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
    TARGET,
    TARGET_INDEX,
    build_logical_program,
    build_native_program,
)

program = build_native_program()
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

logical_program = build_logical_program()
logical_state = (
    simulator.run(
        logical_program,
        shots=0,
        result_config={"counts": False, "final_state": True},
    )
    .result()
    .get_statevector()
)
overlap = np.vdot(state, logical_state)

assert np.isclose(probabilities.sum(), 1.0)
assert np.argmax(probabilities) == TARGET_INDEX
assert np.isclose(probabilities[TARGET_INDEX], 0.9453125, atol=1e-12)
assert np.isclose(abs(overlap), 1.0, atol=1e-12)
assert np.allclose(logical_state, overlap * state, atol=1e-12)

print(f"General Simulator P({TARGET}) = {probabilities[TARGET_INDEX]:.8%}")
circuit_figure = plt.figure(CIRCUIT_FIGURE, figsize=(13.0, 3.2), facecolor="white")
circuit_axis = circuit_figure.add_subplot()
logical_program.draw(ax=circuit_axis, **PROGRAM_DRAW_STYLE)
style_program_figure(circuit_figure, circuit_axis)
draw_distribution(GENERAL_FIGURE, probabilities)
if __name__ == "__main__":
    plt.show()
