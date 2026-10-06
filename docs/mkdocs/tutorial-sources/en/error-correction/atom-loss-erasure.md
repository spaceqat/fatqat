---
title: "Atom loss as an erasure in a repetition code"
description: "Run a repetition-code memory on AtomArraySimulator with atom loss, see what the parity checks reveal about a lost atom, and measure how much the erasure flag in the final readout is worth to a simple decoder."
icon: material-atom
figure_alts:
  - "Logical error rate versus per-gate loss probability for distance-3 and distance-5 repetition codes, decoded by a majority vote over the final readout with and without the erasure flag. Simulated points follow the exact curves, and the curves that use the flag fall much more steeply at each distance."
---


# Atom loss as an erasure in a repetition code


A neutral-atom processor can lose an atom from its trap. The atom takes its
quantum state with it, but the loss is not invisible: when the array is imaged,
the trap is empty. On
[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] an empty site
measures as the digit `2`, distinct from `0` and `1`. An error whose location is
known in this way is called an *erasure*.

Erasures are easier to correct than errors at unknown locations. A code of
distance $d$ corrects any $\lfloor (d-1)/2 \rfloor$ errors at unknown locations,
but up to $d-1$ erasures. This tutorial shows what that means in a small
memory experiment. We run a bit-flip repetition code, lose atoms during its
entangling gates, and look at the loss from two sides: what the parity checks
reveal while the circuit runs, and what the erasure flag in the final readout is
worth to a decoder.

## Imports and settings

```python
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

ROUNDS = 3
SHOTS = 20_000
SEED = 7
```

## A repetition code with reloaded ancillas

The distance-$d$ repetition code stores one bit in $d$ data atoms,
$|0_L\rangle = |0\cdots0\rangle$ and $|1_L\rangle = |1\cdots1\rangle$. Between
neighbouring data atoms sits an ancilla that measures the parity
$Z_j Z_{j+1}$: prepared in $|+\rangle$, it picks up a phase from a `CZ` with
each neighbour, and a final Hadamard turns that phase into a bit. A round of
these measurements is a *syndrome*. For an intact code it reports even parity
everywhere.

The atom array accepts only `RX`, `RY`, `RZ`, and `CZ`, and a `CZ` needs its
two atoms to be paired first, as in the
[eight-atom GHZ tutorial](atom-array-ghz8.md). Each round pairs every ancilla
with its left data atom, applies the `CZ` gates, and then does the same with the
right neighbours. The pairs within a layer are disjoint, so the gates of a layer
can run in parallel.

Every site starts empty. [`Put`][fatqat.operations.Put] loads the data atoms
once and the ancillas at the start of every round. Because `Put` leaves an
occupied site unchanged, it only refills an ancilla lost in an earlier round.
Each round writes its syndrome to its own classical register, and the data
atoms are read out at the end.

```python
def native_h(program: fq.Program, target) -> None:
    """Hadamard in the native gate set, up to a global phase."""
    program.add(ops.RZ(np.pi), target)
    program.add(ops.RY(np.pi / 2), target)


def build_repetition_program(distance: int, logical: int) -> fq.Program:
    """Prepare |logical>, run ROUNDS syndrome rounds, and read out the data."""
    data = fq.QuantumRegister(distance, name="data")
    ancilla = fq.QuantumRegister(distance - 1, name="ancilla")
    syndromes = [
        fq.ClassicalRegister(distance - 1, name=f"round{r}") for r in range(ROUNDS)
    ]
    readout = fq.ClassicalRegister(distance, name="readout")
    program = fq.Program([data, ancilla], [*syndromes, readout])

    program.add(ops.Put, data.all())
    if logical:
        for i in range(distance):
            program.add(ops.RX(np.pi), data[i])

    checks = range(distance - 1)
    for syndrome in syndromes:
        program.add(ops.Put, ancilla.all())  # refill any ancilla lost earlier
        for j in checks:
            native_h(program, ancilla[j])
        for side in (0, 1):  # left neighbours, then right neighbours
            pairs = [(ancilla[j], data[j + side]) for j in checks]
            for pair in pairs:
                program.add(ops.Pair, pair)
            for pair in pairs:
                program.add(ops.CZ, pair)
            for pair in pairs:
                program.add(ops.Unpair, pair)
        for j in checks:
            native_h(program, ancilla[j])
        program.measure(
            tuple(ancilla[j] for j in checks), tuple(syndrome[j] for j in checks)
        )
        for j in checks:
            program.add(ops.Reset, ancilla[j])

    program.measure(
        tuple(data[i] for i in range(distance)),
        tuple(readout[i] for i in range(distance)),
    )
    return program


ideal_counts = (
    fq.simulator.AtomArraySimulator()
    .run(
        build_repetition_program(3, logical=1),
        shots=100,
        simulation_config={"seed": SEED},
    )
    .result()
    .get_counts()
)
print(ideal_counts)
```

Count keys list the classical registers in declaration order: three rounds of
two syndrome bits, then the three data bits. Without noise every check reports
even parity and the data atoms read `111`.

## Lose atoms during the entangling gates

Attach [`Loss`][fatqat.noise.Loss] to `CZ`. The backend samples a loss
independently for each of the two atoms after every `CZ`, so an atom that takes
part in more gates is more likely to be lost. A lost atom stays lost: later
gates on its site do nothing, and its final readout is `2`.

```python
def loss_backend(p_loss: float) -> fq.simulator.AtomArraySimulator:
    noise = fq.NoiseModel()
    noise.add(fq.noise.Loss(p=p_loss), operation=ops.CZ)
    return fq.simulator.AtomArraySimulator(noise=noise)


lossy_counts = (
    loss_backend(0.05)
    .run(
        build_repetition_program(3, logical=1),
        shots=SHOTS,
        simulation_config={"seed": SEED},
    )
    .result()
    .get_counts()
)
print("Most frequent outcomes with 5% loss per atom per CZ:")
for key, n in sorted(lossy_counts.items(), key=lambda item: -item[1])[:6]:
    print(f"  {key}: {n}")
```

A `2` among the syndrome bits is a lost ancilla. It is flagged in the round in
which it happens, since that round's parity is missing, and the next round's
`Put` loads a fresh ancilla. A lost data atom is different. Data atoms are not
imaged until the end, so the `2` of a lost data atom appears only in the final
readout, and nothing in the record says when the atom was lost.

## What a parity check sees of a lost atom

A `CZ` with an empty trap applies no phase, exactly as an atom in $|0\rangle$
would. A check next to a lost data atom therefore measures its surviving
neighbour alone: instead of $Z_j Z_{j+1}$ it measures a single $Z$. The next
cell selects shots in which exactly one data atom and no ancilla was lost, and
counts how often the last syndrome round shows an odd parity.

```python
def flagged_fraction(logical: int, distance: int = 3) -> float:
    counts = (
        loss_backend(0.05)
        .run(
            build_repetition_program(distance, logical),
            shots=SHOTS,
            simulation_config={"seed": SEED},
        )
        .result()
        .get_counts_as_tuples()
    )
    selected = flagged = 0
    for key, n in counts.items():
        syndromes = np.array(key[:-distance]).reshape(ROUNDS, distance - 1)
        readout = key[-distance:]
        if readout.count(2) != 1 or np.any(syndromes == 2):
            continue
        selected += n
        flagged += n * bool(syndromes[-1].any())
    return flagged / selected


for logical in (0, 1):
    print(
        f"logical {logical}: last round shows odd parity in "
        f"{flagged_fraction(logical):.1%} of shots with one lost data atom"
    )
```

For $|0_L\rangle$ the checks never fire, because the lost atom looks like the
$|0\rangle$ atoms around it. For $|1_L\rangle$ they fire in most shots; in the
rest the atom was lost after its last `CZ`, too late for any check to notice.
A lost atom is therefore not a Pauli error: what the checks show depends on the
state the atom held.

This also means the syndrome carries information about the logical value
itself. In the repetition code every single-atom $Z$ is a logical operator, so
a check that has lost one of its atoms measures the logical bit. Codes such as
the surface code do not have this property, because no single-atom $Z$ is a
logical operator there. Keep this in mind below: it is a feature of this small
code, not of atom loss in general.

## Decoders and logical error rates

A *decoder* maps the recorded outcomes of a shot to a guess of the stored
logical bit. Its *logical error rate* (LER) is the probability that the guess is
wrong. The LER is a property of the decoder as much as of the noise: the same
shots decoded two ways give two error rates.

This tutorial uses the simplest decoder for a repetition code: a majority vote
over the final data readout. For the repetition code this is the
minimum-weight correction of the last syndrome, which the final readout
determines. It ignores the syndrome history, so it is not the best possible
decoder, even without loss. What it offers here is a controlled comparison.
We run the same vote in two versions that differ only in whether they use the
erasure flag:

- **Loss-unaware.** The vote cannot tell an empty trap from an atom, as if the
  loss went undetected. Each `2` enters the vote as a random bit.
- **Erasure-aware.** The vote drops every `2` and counts only the surviving
  atoms. A tie, or a shot with no survivors, is decided by a coin flip.

Both versions see the same shots, so the difference between their error rates
is the value of the erasure flag to this decoder.

```python
def logical_error_rates(counts, distance: int, logical: int, rng) -> tuple:
    """Return the (loss-unaware, erasure-aware) logical error rates."""
    unaware = aware = 0.0
    for key, n in counts.items():
        readout = np.array(key[-distance:])
        lost = readout == 2
        ones = np.count_nonzero(readout == 1)
        zeros = np.count_nonzero(readout == 0)

        # Loss-unaware: each lost atom adds a random bit to the vote.
        random_ones = rng.binomial(np.count_nonzero(lost), 0.5, size=n)
        decoded = (ones + random_ones) > distance / 2
        unaware += np.count_nonzero(decoded != logical)

        # Erasure-aware: vote over the survivors; a tie is a coin flip.
        if ones == zeros:
            aware += n / 2
        elif (ones > zeros) != logical:
            aware += n
    total = sum(counts.values())
    return unaware / total, aware / total
```

When loss is the only noise, every surviving atom reads the stored value. The
erasure-aware vote then fails only when all $d$ data atoms are lost and it has
to guess. No decoder that sees only the final readout can do better, since such
a shot contains no information about the logical bit. A decoder that also reads
the syndrome history can do better, because of the leak described in the
previous section, so this is not a lower bound on the LER of the experiment.

The two error rates also have exact values under pure loss. Data atom $i$ takes
part in $n_i$ `CZ` gates: $R$ at either end of the chain and $2R$ in between,
for $R$ rounds. It is lost with probability $q_i = 1-(1-p)^{n_i}$,
independently of the others, and each decoder fails on a known set of loss
patterns. The loss-unaware vote fails when at least $(d+1)/2$ of the random
bits that replace lost atoms are wrong. The erasure-aware vote fails with
probability $1/2$ when every atom is lost.

```python
def exact_error_rates(distance: int, p_loss: float) -> tuple:
    gates_per_atom = np.full(distance, 2 * ROUNDS)
    gates_per_atom[[0, -1]] = ROUNDS
    q = 1 - (1 - p_loss) ** gates_per_atom
    majority = (distance + 1) // 2

    unaware = aware = 0.0
    for pattern in itertools.product((False, True), repeat=distance):
        lost = np.array(pattern)
        probability = np.prod(np.where(lost, q, 1 - q))
        k = int(lost.sum())
        wrong = sum(math.comb(k, w) for w in range(majority, k + 1)) / 2**k
        unaware += probability * wrong
        if k == distance:
            aware += probability / 2
    return unaware, aware
```

## Logical error rate versus loss rate

Sweep the loss probability for distances 3 and 5, and decode each set of shots
both ways.

```python
distances = (3, 5)
loss_rates = np.geomspace(0.02, 0.2, 6)
rng = np.random.default_rng(SEED)

sampled = {}
for distance in distances:
    program = build_repetition_program(distance, logical=1)
    for p_loss in loss_rates:
        counts = (
            loss_backend(p_loss)
            .run(program, shots=SHOTS, simulation_config={"seed": SEED})
            .result()
            .get_counts_as_tuples()
        )
        sampled[distance, p_loss] = logical_error_rates(counts, distance, 1, rng)

print("   d  p_loss  loss-unaware  erasure-aware")
for (distance, p_loss), (unaware, aware) in sampled.items():
    print(f"  {distance}  {p_loss:6.3f}  {unaware:12.4f}  {aware:13.5f}")
```

```python
fine_rates = np.geomspace(loss_rates[0], loss_rates[-1], 100)
colors = {3: "tab:blue", 5: "tab:orange"}

figure, axis = plt.subplots(figsize=(7, 4.5))
for distance in distances:
    exact = np.array([exact_error_rates(distance, p) for p in fine_rates])
    axis.plot(fine_rates, exact[:, 0], color=colors[distance], linestyle="--")
    axis.plot(fine_rates, exact[:, 1], color=colors[distance])
    for column, marker, label in ((0, "o", "loss-unaware"), (1, "s", "erasure-aware")):
        rates = np.array([sampled[distance, p][column] for p in loss_rates])
        seen = rates > 0
        axis.plot(
            loss_rates[seen],
            rates[seen],
            marker,
            color=colors[distance],
            markerfacecolor="none" if column == 0 else colors[distance],
            label=f"d={distance}, {label}",
        )
axis.set(
    xscale="log",
    yscale="log",
    xlabel="Loss probability per atom per CZ",
    ylabel="Logical error rate",
    title="Majority vote over the final readout",
)
axis.legend(fontsize="small")
figure.tight_layout()
plt.show()
```

Markers are sampled error rates; dashed and solid lines are the exact
loss-unaware and erasure-aware rates. Points with no failures in the sampled
shots are left out, which is why the distance-5 erasure-aware curve has no
markers at the lowest loss rates.

At small $p$ each curve becomes a straight line whose slope counts the losses
needed to cause a logical error. The loss-unaware vote fails once
$(d+1)/2$ lost atoms happen to contribute the wrong bit, so its error rate
scales as $p^{(d+1)/2}$. The erasure-aware vote fails only when all $d$ atoms
are lost, so it scales as $p^d$. These are the two numbers from the
introduction, $\lfloor (d-1)/2 \rfloor + 1$ unknown errors against $d$
erasures. Going from distance 3 to 5 gains the loss-unaware vote one power of
$p$ and the erasure-aware vote two.

## Loss together with bit flips

Atoms that stay in their traps can still flip. Add a 2% `X` error on the data
atom of every `CZ`, keep 5% loss, and decode again. The surviving atoms are no
longer always right, so the erasure-aware vote can now fail with atoms left to
vote, but it still removes the lost atoms instead of letting them add random
bits.

```python
mixed_noise = fq.NoiseModel()
mixed_noise.add(fq.noise.Loss(p=0.05), operation=ops.CZ)
mixed_noise.add(
    fq.noise.PauliChannel({"X": 0.02}), operation=ops.CZ, target_positions=1
)
mixed_backend = fq.simulator.AtomArraySimulator(noise=mixed_noise)

print("   d  loss-unaware  erasure-aware")
for distance in distances:
    counts = (
        mixed_backend.run(
            build_repetition_program(distance, logical=1),
            shots=SHOTS,
            simulation_config={"seed": SEED},
        )
        .result()
        .get_counts_as_tuples()
    )
    unaware, aware = logical_error_rates(counts, distance, 1, rng)
    print(f"  {distance}  {unaware:12.4f}  {aware:13.4f}")
```

## Takeaways and next steps

`AtomArraySimulator` reports a lost atom as the digit `2`, which turns atom
loss into an erasure. For a majority vote over the final readout, using that
flag changes the scaling of the logical error rate from $p^{(d+1)/2}$ to
$p^d$. The comparison holds the decoder fixed and so isolates the value of the
flag; it does not show the best error rate the experiment allows.

While the circuit runs, loss is harder to read. A lost ancilla is flagged in its
own round, but a lost data atom shows up in the checks only as a changed
parity, which depends on the state the atom held. A decoder that uses the
syndrome history must therefore treat a data-atom loss as a set of possible
error patterns, one for each time at which the atom may have been lost. Such
decoders go beyond this tutorial.

To explore further, attach `Loss` to `Pair` and `Unpair` to model the cost of
transport, as in the [eight-atom GHZ tutorial](atom-array-ghz8.md). You can
also raise `ROUNDS` to see loss accumulate over a longer memory, or replace the
majority vote with a decoder that reads the syndrome history and compare its
error rate with the curves above.
