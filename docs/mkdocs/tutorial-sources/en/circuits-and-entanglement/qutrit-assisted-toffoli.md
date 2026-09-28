---
title: "Build a Toffoli gate by borrowing a third level"
description: "Use a qubit–qutrit–qubit register to construct Toffoli with three qubit–qutrit interactions, then verify its relative phases and return from the borrowed level."
figure_alts:
  - "One simulated input follows 110, 120, 121, and 111 at the input, controlled level exchange, target flip, and restoration checkpoints; the middle system's level-2 population is zero, one, one, and zero."
  - "Transition probabilities from eight logical inputs to all twelve output states: only 110 and 111 exchange places, and the four orange-outlined output rows with middle level 2 have zero population."
---

# Build a Toffoli gate by borrowing a third level

In this tutorial, we build a Toffoli gate with three qubit–qutrit interactions
and verify its action on quantum states. The middle system is a qutrit: it
has levels $|0\rangle$, $|1\rangle$, and $|2\rangle$. We use its first two
levels as a logical qubit and borrow the third during the gate. The final
checks will show that we recover the required logical output, including its
relative phases, and empty the borrowed level.

You should be familiar with qubit basis states, superpositions, and basic
single-qubit gates. Here we use ideal gate simulation to study a register
with different local dimensions.

A Toffoli gate has two controls, $a$ and $b$, and a target, $t$. It flips
the target exactly when both controls are $1$:

$$
|a,b,t\rangle \longmapsto |a,b,t\oplus(a b)\rangle,
$$

where $\oplus$ means addition modulo two. Starting the target at zero
therefore computes the logical AND of the controls without changing their
values. The gate exchanges $|110\rangle$ and $|111\rangle$ and leaves the
other six logical basis states unchanged.

We give the middle control, $b$, access to $|2\rangle$ and construct the
gate in three steps:

1. **Borrow the third level.** When the input has $a=1$ and $b=1$, move
   $b$ from $|1\rangle$ to $|2\rangle$.
2. **Flip the target.** Apply $X$ to $t$ when $b=2$.
3. **Return the control.** Undo the first step so that $b$ has its
   original value.

Each step uses a short sequence of gates, which we call a block. At the
end of the first block, occupation of level 2 records that both input
controls were $1$. We can then condition the target flip on this one level
and undo the temporary encoding, using the same three subsystems throughout.

## Set up the mixed register

We give each control and the target a named one-slot register. The registers
`a` and `t` use the default qubit dimension, while `dim=3` makes `b` a
qutrit. We address their slots as `a[0]`, `b[0]`, and `t[0]`, following the
[mixed-register guide](../guide/program.md#mix-qubits-and-qutrits).

The joint state has $2\times3\times2=12$ amplitudes. Eight basis states
have `b=0` or `b=1`; these form the logical subspace on which we want
Toffoli to act. The remaining four have `b=2` and are available during the
gate.

```python
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

a = fq.QuantumRegister(1, name="a")
b = fq.QuantumRegister(1, name="b", dim=3)
t = fq.QuantumRegister(1, name="t")
dims = (2, 3, 2)
basis_labels = ["".join(map(str, digits)) for digits in np.ndindex(dims)]
```

Returned states follow register declaration order, with the last subsystem
changing fastest. For these dimensions, $|a,b,t\rangle$ has flat index
$6a+2b+t$. For example, $|110\rangle$ is at index $8$, and $|120\rangle$
is at index $10$. This differs from the three-qubit index because the middle
system has three levels.

The helper below creates a basis state with unit amplitude at the requested
index and zero everywhere else. We will use it both to choose inputs and to
write down the states we expect from the gate.

```python
def basis_state(a_level, b_level, t_level):
    state = np.zeros(12, dtype=complex)
    state[6 * a_level + 2 * b_level + t_level] = 1
    return state
```

## Exchange levels 1 and 2 when a=1

The first block must distinguish inputs with both controls equal to $1$.
We do this with a controlled level exchange: exchange the middle system's
levels 1 and 2 when `a=1`, and leave them alone when `a=0`. Level `b=0`
stays unchanged in either case.

We can build the controlled exchange from two rotations and a controlled
phase. `SubspaceRY(theta, (1, 2))` applies the usual qubit $R_y$ rotation
within the ordered pair $(|1\rangle,|2\rangle)$ of `b`, leaving its
$|0\rangle$ level untouched.

Between the rotations, we use `CClock(1)` to change a sign when `a=1`.
Its second operand's dimension sets the phase's root of unity. We put the
qutrit first and the qubit second, `(b[0], a[0])`, to obtain the binary
phases $\pm1$:

$$
|b,a\rangle \longmapsto (-1)^{b a}|b,a\rangle.
$$

This formula uses the pair's operand order, $(b,a)$; the full register
remains in $(a,b,t)$ order. Putting the qutrit second would introduce third
roots of unity and change the gate.

When `a=0`, the phase is always $+1$ and the two opposite rotations cancel.
When `a=1`, level `b=1` gains a minus sign, while levels 0 and 2 keep
their phase. In the selected pair, placing this sign change between the
rotations gives

$$
R_y(-\pi/2)
\begin{pmatrix}-1&0\\0&1\end{pmatrix}
R_y(\pi/2)
=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

The matrix on the right exchanges levels 1 and 2 with no extra phase.
Matrix products act from right to left, so we apply the positive rotation
first. The list below keeps the operations and their targets in execution
order:

```python
controlled_exchange = [
    (ops.SubspaceRY(np.pi / 2, (1, 2)), b[0]),
    (ops.CClock(1), (b[0], a[0])),
    (ops.SubspaceRY(-np.pi / 2, (1, 2)), b[0]),
]
```

For a logical input, this block moves $|11t\rangle$ to $|12t\rangle$ and
leaves the other control combinations unchanged. After this exchange,
`b=2` tells us that both controls were $1$ at the input. Repeating the block
swaps the levels back, so the controlled exchange is its own inverse. The
[qudit-gate reference](../api/operations/qudit-gates.md) gives the full
rotation and controlled-phase conventions.

## Flip the target when b=2

The first block has encoded the two-control condition in `b=2`. We now
need a target flip conditioned on that level, while leaving `t` unchanged
for `b=0` and `b=1`.

We again put the qubit second in the controlled phase, now using
`(b[0], t[0])`. It applies $Z$ to `t` when `b=1` and the identity when
`b=0` or `b=2`. Placing `H` before and
after it gives a target flip for `b=1`, because $H Z H=X$. For the other
two levels, the Hadamards cancel: $H I H=I$.

To select states that enter with `b=2`, we exchange levels 1 and 2 using
`SwapLevels(1, 2)` before the `H`, `CClock`, `H` sequence and exchange them
again afterward. Tracking each possible value of `b` gives:

| `b` at the start of this block | After the first swap | Effect of `H`, `CClock`, `H` on `t` | After the second swap |
|---|---|---|---|
| 0 | 0 | Unchanged | 0 |
| 1 | 2 | Unchanged | 1 |
| 2 | 1 | Flipped | 2 |

The two swaps return `b` to the value it had at the start of this block.
Only the target of a state that entered with `b=2` is flipped.

```python
flip_on_level_2 = [
    (ops.SwapLevels(1, 2), b[0]),
    (ops.H, t[0]),
    (ops.CClock(1), (b[0], t[0])),
    (ops.H, t[0]),
    (ops.SwapLevels(1, 2), b[0]),
]
```

## Restore the middle control

For input $|110\rangle$, the first two blocks produce $|121\rangle$.
The target has the correct value, $t=1$, but the middle control is still
at $b=2$. Toffoli must leave both controls at their input values, so the
required output is $|111\rangle$.

We finish the gate by repeating `controlled_exchange`. Since `a=1`, it
exchanges `b`'s levels 1 and 2 again, taking $|121\rangle$ to $|111\rangle$
without changing the target. Inputs with `a=0` or `b=0` remain unchanged
by this last block.

This reversal of the temporary encoding is called uncomputation. We undo
the encoding with coherent gates, preserving the relative amplitudes of
input branches. After tracing one basis input, we will check that restoration
also works for a superposition.

## Assemble and run the program

We have the three blocks needed for Toffoli. Joining them in order gives
the complete gate; keeping their instruction lists also lets us stop at
block boundaries to inspect the temporary encoding. The helper
`make_program` turns an instruction list into a `Program` through `Program.add`.

```python
sequence = controlled_exchange + flip_on_level_2 + controlled_exchange


def make_program(instructions):
    program = fq.Program([a, b, t])
    for operation, targets in instructions:
        program.add(operation, targets)
    return program


toffoli_program = make_program(sequence)
```

To follow the borrowed level, we need the complex state after each block.
The `statevector` method returns this vector. In `evolve`, `initial_state`
chooses the input, and `result_config` requests the final state and disables
counts. These ideal runs contain no measurements or noise, so they have no
sampling uncertainty and need no random seed. We can use the same helper
for the full gate or a prefix of its instructions.

```python
state_backend = fq.simulator.Simulator(method="statevector", runtime="numpy")


def evolve(instructions, initial_state):
    result = state_backend.run(
        make_program(instructions),
        initial_state=initial_state,
        result_config={"counts": False, "final_state": True},
    ).result()
    return result.get_statevector()
```

## Follow one input through the borrowed level

We start with $|110\rangle$: both controls are $1$, and the target is $0$.
The controlled exchange moves the middle control to level 2, the middle
block flips the target, and restoration brings the control back. The
expected states at the block boundaries are

$$
|110\rangle
\xrightarrow{\text{exchange}}|120\rangle
\xrightarrow{\text{flip}}|121\rangle
\xrightarrow{\text{restore}}|111\rangle.
$$

The boundaries occur after 0, 3, 8, and 11 instructions. Each prefix run
starts from the same $|110\rangle$ input; its returned state is compared
directly with the expected complex vector.

We also compute the probability of finding `b` in level 2. Reshaping a
state to `(2, 3, 2)` gives one axis per subsystem. The slice `[:, 2, :]`
selects every amplitude with `b=2`, regardless of `a` or `t`, and summing
their squared magnitudes gives the borrowed-level population.

```python
checkpoints = [0, 3, 8, 11]
stage_names = ["Input", "Exchange", "Flip target", "Restore"]
branch_states = [evolve(sequence[:stop], basis_state(1, 1, 0)) for stop in checkpoints]
expected_digits = [(1, 1, 0), (1, 2, 0), (1, 2, 1), (1, 1, 1)]

for state, digits in zip(branch_states, expected_digits):
    np.testing.assert_allclose(state, basis_state(*digits), atol=1e-12, rtol=0)

branch_labels = [basis_labels[np.argmax(np.abs(state))] for state in branch_states]
borrowed_populations = [
    np.sum(np.abs(state.reshape(dims)[:, 2, :]) ** 2) for state in branch_states
]

for name, label, population in zip(stage_names, branch_labels, borrowed_populations):
    print(f"{name:>11}: |{label}>, P(b=2) = {population:.3e}")
```

The printed populations follow $0,1,1,0$ to numerical precision. For this
input, the middle control occupies level 2 after the exchange and target
flip, then returns to level 1. The figure pairs each population with its
state label so we can inspect both the control's return and the target's flip.

```python
figure, axis = plt.subplots(figsize=(7.6, 4.2))
positions = np.arange(len(checkpoints))
axis.bar(positions, borrowed_populations, width=0.5, color="#c97920")
axis.scatter(positions, borrowed_populations, color="#805014", zorder=3)
for position, label in zip(positions, branch_labels):
    axis.text(
        position, 1.19, rf"$|{label}\rangle$", ha="center", va="center", fontsize=16
    )
for position in positions[:-1]:
    axis.annotate(
        "",
        xy=(position + 0.72, 1.19),
        xytext=(position + 0.28, 1.19),
        arrowprops={"arrowstyle": "->", "color": "0.45"},
    )
axis.set(
    xticks=positions,
    xticklabels=stage_names,
    yticks=[0, 0.5, 1],
    ylim=(-0.05, 1.4),
    xlabel="Circuit checkpoint",
    ylabel=r"Borrowed-level population $P(b=2)$",
    title=r"Borrow, flip, return: input $|110\rangle$",
)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
plt.show()
```

The horizontal axis marks block boundaries, not elapsed time. Rotations
and swaps can also populate level 2 within a block; each bar describes the
state at its checkpoint. The tiny final residual is consistent with
floating-point arithmetic.

## Check a superposition

The trace confirms the route through level 2 for one input. We now combine
two branches: after the controlled exchange, one occupies level 2 and the
other remains in the logical subspace. To prepare them, we apply `H` followed
by `S` to `a`, giving
$(|0\rangle+i|1\rangle)/\sqrt2$. A `Shift(1)` puts `b` in level 1,
while `t` starts at zero. The factor $i$ gives us a relative phase to track.

This input contains two branches. The $|010\rangle$ branch should stay
unchanged because its first control is zero. The $|110\rangle$ branch
should undergo the exchange, flip, and restoration we just traced. The
full gate should therefore give

$$
\frac{|010\rangle+i|110\rangle}{\sqrt2}
\longmapsto
\frac{|010\rangle+i|111\rangle}{\sqrt2}.
$$

After the first block, only the second branch occupies level 2. Restoration
must bring it back without altering its relative factor $i$. We keep
preparation separate from the gate: first create `coherent_input`, then
pass it to a run of `sequence`. The assertions check both the expected
complex output and the absence of any remaining borrowed-level amplitude.

```python
preparation = [
    (ops.H, a[0]),
    (ops.S, a[0]),
    (ops.Shift(1), b[0]),
]
coherent_input = evolve(preparation, basis_state(0, 0, 0))
expected_input = (basis_state(0, 1, 0) + 1j * basis_state(1, 1, 0)) / np.sqrt(2)
np.testing.assert_allclose(coherent_input, expected_input, atol=1e-12, rtol=0)

coherent_output = evolve(sequence, coherent_input)
expected_output = (basis_state(0, 1, 0) + 1j * basis_state(1, 1, 1)) / np.sqrt(2)
np.testing.assert_allclose(coherent_output, expected_output, atol=1e-12, rtol=0)
np.testing.assert_allclose(
    coherent_output.reshape(dims)[:, 2, :], 0, atol=1e-12, rtol=0
)

print("Nonzero output amplitudes:")
for label, amplitude in zip(basis_labels, coherent_output):
    if abs(amplitude) > 1e-12:
        print(f"  |{label}>: {amplitude:.6f}")
print(
    "Final borrowed-level population:",
    np.sum(np.abs(coherent_output.reshape(dims)[:, 2, :]) ** 2),
)
```

The two printed amplitudes are $1/\sqrt2$ and $i/\sqrt2$, as expected.
Both branches end with `b=1`, while `a` and `t` are entangled with a
relative phase of $i$ between their $|00\rangle$ and $|11\rangle$ components.

Measuring these branches in the computational basis would give equal
probabilities. Replacing $i$ by $-i$ would give the same probabilities but
a different quantum state. Comparing complex amplitudes lets us detect
that phase error, which a probability-only check would miss.

## Check all eight logical inputs

The superposition check tests coherence for two input branches. We can
cover the whole logical subspace by comparing the output for each of its
eight basis states, including the complex amplitudes. The `unitary`
simulation method gives us all those outputs in the full $12\times12$
matrix of the gate. We run `toffoli_program`, which contains only the gate.

Each column is the output state for one basis input. Entry $(i,j)$ is the
amplitude for output state $i$ when the input is state $j$. Both axes use
the [result conventions](../api/result.md#ordering-and-mutable-values)
introduced above. For example, the column for $|110\rangle$ should have
unit amplitude in the row for $|111\rangle$.

The logical states are interleaved with the borrowed-level states in this
ordering. We select the eight logical rows and columns to obtain the gate's
logical matrix. We also select the four borrowed-level output rows to
check for leakage: amplitude left outside the logical subspace at the end
of the gate.

For comparison, we build a reference $8\times8$ Toffoli matrix from its
Boolean rule. Here the indices use the three-bit formula $4a+2b+t$.
For each input column, `t_bit ^ (a_bit & b_bit)` gives the output target
bit; we put a one in the corresponding output row and leave the other
entries zero.

```python
unitary = (
    fq.simulator.Simulator(method="unitary", runtime="numpy")
    .run(toffoli_program, result_config={"counts": False, "final_state": True})
    .result()
    .get_unitary()
)

logical_indices = np.array([0, 1, 2, 3, 6, 7, 8, 9])
borrowed_indices = np.array([4, 5, 10, 11])
expected_ccx = np.zeros((8, 8), dtype=complex)
for column, (a_bit, b_bit, t_bit) in enumerate(np.ndindex(2, 2, 2)):
    output_bit = t_bit ^ (a_bit & b_bit)
    row = 4 * a_bit + 2 * b_bit + output_bit
    expected_ccx[row, column] = 1

logical_unitary = unitary[np.ix_(logical_indices, logical_indices)]
leakage_block = unitary[np.ix_(borrowed_indices, logical_indices)]
np.testing.assert_allclose(logical_unitary, expected_ccx, atol=1e-12, rtol=0)
np.testing.assert_allclose(leakage_block, 0, atol=1e-12, rtol=0)

print("Maximum complex logical error:", np.max(np.abs(logical_unitary - expected_ccx)))
print("Maximum final leakage amplitude:", np.max(np.abs(leakage_block)))
```

Both printed errors should be near floating-point precision. The first
assertion compares the logical matrix with Toffoli, including every complex
phase. The second requires the amplitudes from logical inputs to level-2
outputs to be zero. These checks use the returned amplitudes without a
phase adjustment or renormalization, so a phase mismatch or amplitude left
in level 2 would remain visible.

Any logical superposition is a linear combination of the eight input
columns. Matching all those columns, including their phases and their
borrowed-level entries, therefore establishes the intended action for
arbitrary logical superpositions to numerical precision.

With the complex matrix checked, we can visualize the truth table by
plotting the squared magnitudes of the eight logical input columns. We keep
all twelve output rows so that any final borrowed-level population would
also appear.

```python
transition_probabilities = np.abs(unitary[:, logical_indices]) ** 2
logical_labels = [basis_labels[index] for index in logical_indices]

figure, axis = plt.subplots(figsize=(7.2, 6.3))
heatmap = axis.imshow(transition_probabilities, cmap="Blues", vmin=0, vmax=1)
axis.set(
    xticks=np.arange(8),
    xticklabels=[rf"$|{label}\rangle$" for label in logical_labels],
    yticks=np.arange(12),
    yticklabels=[rf"$|{label}\rangle$" for label in basis_labels],
    xlabel="Logical input",
    ylabel="Output in the (2, 3, 2) register",
    title="Toffoli action in the larger state space",
)
for row in borrowed_indices:
    axis.axhspan(
        row - 0.5, row + 0.5, facecolor="none", edgecolor="#c97920", linewidth=1.5
    )
    axis.get_yticklabels()[row].set_color("#996019")
figure.colorbar(heatmap, ax=axis, label="Transition probability", shrink=0.8)
figure.tight_layout()
plt.show()
```

Read the heatmap one column at a time. Inputs `110` and `111` exchange
places, while the other six logical inputs return to themselves. The four
orange-outlined rows correspond to `b=2` and have zero population to numerical
precision for all eight inputs. The figure shows the correct truth table
and return to the logical subspace; the complex comparisons above supply
the phase-sensitive verification.

## Count the interactions

Having verified the gate, we can count the operations needed to construct
it. Each `CClock` couples the qutrit `b` to one qubit, either `a` or `t`.
The other gates act on one subsystem at a time. We count these groups in
the same `sequence` we simulated, using each operation's `num_subsystems`
value. The count covers the gate itself; input preparation is separate.

```python
interaction_count = sum(operation.num_subsystems == 2 for operation, _ in sequence)
single_gate_count = sum(operation.num_subsystems == 1 for operation, _ in sequence)
print(f"Primitive instructions: {len(sequence)}")
print(f"Qubit–qutrit interactions: {interaction_count} (all CClock)")
print(f"Single-qubit or single-qutrit gates: {single_gate_count}")
```

There are eleven primitive instructions: three qubit–qutrit interactions,
two single-qubit gates (`H` on `t`), and six single-qutrit gates (four
rotations and two level swaps on `b`). Each of the three blocks uses one
interaction, with local gates selecting how it acts on the logical states.

In [*Efficient Toffoli Gates Using Qudits*](https://arxiv.org/abs/0806.0654),
Ralph, Resch, and Gilchrist showed how access to a third level reduces
Toffoli to three controlled-sign interactions. This FatQat construction
follows that principle with a different gate sequence. To compare physical
speed or fidelity, we would also need the durations and errors of the
rotations and interactions.

## Exercise: leave out restoration

Run the following code after the tutorial cells to omit the restoration
block. It applies the first two blocks to $|110\rangle$ and prints the
output amplitudes and borrowed-level population. Predict the output state
before running it, then compare it with the required $|111\rangle$.

~~~python
without_restoration = controlled_exchange + flip_on_level_2
unfinished_state = evolve(without_restoration, basis_state(1, 1, 0))
for label, amplitude in zip(basis_labels, unfinished_state):
    if abs(amplitude) > 1e-12:
        print(f"|{label}>: {amplitude:.6f}")
print("P(b=2):", np.sum(np.abs(unfinished_state.reshape(dims)[:, 2, :]) ** 2))
~~~

Next, replace `basis_state(1, 1, 0)` in the exercise with `coherent_input`.
Predict which branch will remain in level 2 and its contribution to
`P(b=2)`, then compare the returned amplitudes with `expected_output`.
Explain why getting the target bit right on each branch is insufficient
to recover the intended quantum state. Finally, use the full `sequence`
and check that the middle control returns to the logical subspace.
