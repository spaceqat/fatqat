---
title: "Split a power grid into islands with QAOA"
description: "Model controlled islanding of the IEEE test systems as a QUBO, run QAOA on FatQat, and repair every measured outcome into a feasible split with pVSQA postprocessing."
icon: material-transmission-tower
figure_alts:
  - "IEEE 14-bus network with the optimal two-island split and its cut lines"
  - "Mean energy and optimal-solution probability against QAOA depth"
  - "Energy distribution of raw QAOA samples compared with repaired ones"
  - "Qubit count per encoding across five IEEE test systems"
---

# Split a power grid into islands with QAOA

When a disturbance threatens to cascade through a transmission network, one
last-resort defense is to stop fighting it and split the grid deliberately.
*Controlled islanding* opens a chosen set of lines so the network falls into
separate islands, each of which can survive on its own. The choice of lines has
to be made in seconds, and it must satisfy several requirements at once:

- **Minimize the disruption.** Every line opened interrupts the power flowing
  on it, so the total flow on the cut lines is the cost to minimize.
- **Respect generator coherency.** After a disturbance, generators separate
  into groups that swing together. Two generators from different groups left in
  one island will pull against each other and trip it. Coherent generators must
  stay together; incoherent ones must be separated.
- **Keep every island viable.** An island needs generation, load, and enough
  buses to be worth operating.
- **Keep every island connected.** A set of buses with no path between them is
  not an island.

That is a combinatorial optimization problem over line-opening decisions, and
it grows exponentially with the network. This tutorial models it as a QUBO,
runs QAOA on it with FatQat, and — crucially — repairs the measured bitstrings
into feasible splits, because a shallow QAOA circuit on a constrained problem
returns mostly garbage without that step.

The mechanics of turning a QUBO into a FatQat program are covered in
[Solve a QUBO with QAOA](qubo-qaoa.md); this tutorial reuses
them and concentrates on the modeling and the postprocessing.

## The networks

Five IEEE test systems are used below. Two are carried in full — the WSCC
9-bus and the IEEE 14-bus — while the larger three appear only in the qubit
accounting at the end. Bus labels are zero-based, so bus 0 is the system's bus
1. Line weights are the steady-state power flows of the reference case, which
is what makes the cut weight the power interrupted by the split.

The coherency groups are the interesting part of the data. On the 14-bus
system, generators 0, 1, and 2 swing together and generators 5 and 7 swing
together, so any acceptable split must put `{0, 1, 2}` on one side and
`{5, 7}` on the other. That single requirement is what stops the answer from
being the cheapest cut in the graph.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import fatqat as fq
import fatqat.operations as ops

SYSTEMS = {
    "9-bus": {
        "edges": [
            (0, 3, 0.8855058426108287),
            (2, 5, 1.0506270719315443),
            (3, 4, 0.3795071371756076),
            (4, 5, 0.7349783693597357),
            (5, 6, 0.29891469957856975),
            (6, 7, 0.9382048153285490),
            (7, 1, 2.0147319144099027),
            (7, 8, 1.0706524421112620),
            (8, 3, 0.5028157475281494),
        ],
        "generators": [0, 1, 2],
        "loads": [4, 5, 7, 8],
        "coherency": [[1, 2], [0]],
        "islands": 2,
    },
    "14-bus": {
        "edges": [
            (0, 1, 154.73), (0, 4, 74.129), (1, 2, 72.076), (1, 3, 55.293),
            (1, 4, 41.064), (2, 3, 23.472), (3, 4, 61.415), (3, 6, 28.074),
            (3, 8, 16.080), (4, 5, 44.087), (5, 10, 7.3256), (5, 11, 7.7502),
            (5, 12, 17.642), (6, 7, 5.7112e-11), (6, 8, 28.074), (8, 9, 5.2211),
            (8, 13, 9.3683), (9, 10, 3.7916), (11, 12, 1.6111), (12, 13, 5.6168),
        ],
        "generators": [0, 1, 2, 5, 7],
        "loads": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
        "coherency": [[0, 1, 2], [5, 7]],
        "islands": 2,
    },
}

for name, spec in SYSTEMS.items():
    spec["buses"] = sorted({b for i, j, _ in spec["edges"] for b in (i, j)})
    spec["lines"] = [(i, j) for i, j, _ in spec["edges"]]
    spec["weight"] = {(i, j): w for i, j, w in spec["edges"]}
    print(
        f"{name}: {len(spec['buses'])} buses, {len(spec['lines'])} lines, "
        f"{len(spec['generators'])} generators, {len(spec['loads'])} loads, "
        f"coherency groups {spec['coherency']}, "
        f"total flow {sum(w for *_, w in spec['edges']):.3f}"
    )


def line_weight(spec, i, j):
    """Return the flow on a line, accepting either endpoint order."""
    return spec["weight"].get((i, j), spec["weight"].get((j, i), 1.0))
```

## Two encodings for the same decision

The textbook encoding gives every bus one binary variable per island:
$y_{i,g} = 1$ when bus $i$ joins island $g$. It handles any number of islands,
but most of its bitstrings are nonsense — a bus can be assigned to two islands,
or to none — so it needs a one-hot penalty $\sum_i (\sum_g y_{i,g} - 1)^2$ to
forbid them, and it costs $n_\text{buses} \times k$ qubits.

When there are exactly two islands, the one-hot rule says $y_{i,1} = 1 -
y_{i,0}$: the second variable carries no information. Keeping one variable per
bus, whose value names the island directly, halves the qubit count *and*
deletes the penalty, because no bitstring can violate a rule the encoding
cannot express. That is the single largest lever available on a near-term
device, and it costs nothing.

Both encodings are built below. The rest of the requirements are the same in
each: they only differ in how "bus $i$ is in island $g$" is written down.

```python
def build_qubo(spec, encoding="binary", penalty=None, balance_weight=None):
    """Return (constant, linear, quadratic, variables) for one islanding model."""
    buses, lines, k = spec["buses"], spec["lines"], spec["islands"]
    # Line weights are physical flows, so they differ by orders of magnitude
    # between systems. Every penalty is stated as a multiple of the heaviest
    # line, which bounds what any single reassignment can save.
    scale = max(line_weight(spec, i, j) for i, j in lines)
    if penalty is None:
        penalty = 2.0 * scale
    if balance_weight is None:
        # Balance only has to forbid a degenerate one-bus island, so it sits
        # far below the hard constraints; heavier, and it overrides the
        # objective it is meant to guard.
        balance_weight = 0.1 * scale
    if encoding == "binary" and k != 2:
        raise ValueError("the binary encoding needs exactly two islands")

    if encoding == "one_hot":
        variables = [f"y[{b},{g}]" for b in buses for g in range(k)]
    else:
        variables = [f"y[{b}]" for b in buses]
    index = {name: position for position, name in enumerate(variables)}

    constant, linear = 0.0, np.zeros(len(variables))
    quadratic: dict[tuple[int, int], float] = {}

    def add_quadratic(a, b, value):
        i, j = index[a], index[b]
        if i == j:
            linear[i] += value          # x * x == x for binary x
            return
        key = (i, j) if i < j else (j, i)
        quadratic[key] = quadratic.get(key, 0.0) + value

    def add_square(terms, offset, weight):
        """Add weight * (sum_k c_k v_k + offset)**2, expanded with x*x = x."""
        nonlocal constant
        constant += weight * offset**2
        for name, c in terms:
            linear[index[name]] += weight * (c * c + 2 * offset * c)
        for a in range(len(terms)):
            for b in range(a + 1, len(terms)):
                add_quadratic(terms[a][0], terms[b][0], 2 * weight * terms[a][1] * terms[b][1])

    def member(bus, island):
        """Terms and offset whose sum is 1 exactly when bus is in island."""
        if encoding == "one_hot":
            return [(f"y[{bus},{island}]", 1.0)], 0.0
        # Under the binary encoding, island 1 is the variable and island 0 is
        # its complement.
        return ([(f"y[{bus}]", 1.0)], 0.0) if island == 1 else ([(f"y[{bus}]", -1.0)], 1.0)

    # Objective: the flow on every cut line.
    for i, j in lines:
        w = line_weight(spec, i, j)
        if encoding == "one_hot":
            constant += w
            for g in range(k):
                add_quadratic(f"y[{i},{g}]", f"y[{j},{g}]", -w)
        else:
            linear[index[f"y[{i}]"]] += w
            linear[index[f"y[{j}]"]] += w
            add_quadratic(f"y[{i}]", f"y[{j}]", -2 * w)

    # Exactly one island per bus. Unnecessary, and skipped, under "binary".
    if encoding == "one_hot":
        for bus in buses:
            add_square([(f"y[{bus},{g}]", 1.0) for g in range(k)], -1.0, penalty)

    # Coherency: groups stay whole, and different groups stay apart.
    groups = spec["coherency"]
    for group in groups:
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                if encoding == "one_hot":
                    for g in range(k):
                        add_square(
                            [(f"y[{group[a]},{g}]", 1.0), (f"y[{group[b]},{g}]", -1.0)],
                            0.0, penalty,
                        )
                else:
                    add_square(
                        [(f"y[{group[a]}]", 1.0), (f"y[{group[b]}]", -1.0)], 0.0, penalty
                    )
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            for i in groups[a]:
                for j in groups[b]:
                    if encoding == "one_hot":
                        for g in range(k):
                            add_quadratic(f"y[{i},{g}]", f"y[{j},{g}]", penalty)
                    else:
                        # 1 - (y_i + y_j - 2 y_i y_j) is 1 when they agree.
                        constant += penalty
                        linear[index[f"y[{i}]"]] -= penalty
                        linear[index[f"y[{j}]"]] -= penalty
                        add_quadratic(f"y[{i}]", f"y[{j}]", 2 * penalty)

    # Balance: islands of equal size. An equality, so it needs no slack qubits,
    # and it rules out the degenerate split that isolates a single bus.
    target = len(buses) / k
    for g in range(k):
        terms, offset = [], 0.0
        for bus in buses:
            bus_terms, bus_offset = member(bus, g)
            terms += bus_terms
            offset += bus_offset
        add_square(terms, offset - target, balance_weight)

    return constant, linear, quadratic, variables


for encoding in ("one_hot", "binary"):
    for name, spec in SYSTEMS.items():
        constant, linear, quadratic, variables = build_qubo(spec, encoding)
        print(f"{name:8} {encoding:8} {len(variables):3d} qubits, {len(quadratic):4d} pair terms")
```

## What the model actually says

With 14 or fewer variables the whole space fits in memory, so the model can be
checked exhaustively before any quantum work. That check is worth doing, and on
this problem it turns up something important.

The energy of a bitstring is only half the story. Decoding it into an island
assignment and running the operating checks — including connectivity, which is
deliberately *not* in the QUBO — is what says whether the split is usable.

```python
def spectrum_of(constant, linear, quadratic, n):
    """Return every assignment's energy, indexed the way FatQat indexes states."""
    codes = np.arange(1 << n)
    columns = [((codes >> (n - 1 - p)) & 1).astype(bool) for p in range(n)]
    energies = np.full(1 << n, constant)
    for position, value in enumerate(linear):
        if value:
            energies[columns[position]] += value
    for (i, j), value in quadratic.items():
        energies[columns[i] & columns[j]] += value
    return energies


def to_bits(index, n):
    return np.array([(index >> (n - 1 - p)) & 1 for p in range(n)], dtype=np.int8)


def connected(members, lines):
    """Return whether the subgraph induced on a bus set is connected."""
    if len(members) <= 1:
        return True
    inside = set(members)
    adjacency = {bus: [] for bus in inside}
    for i, j in lines:
        if i in inside and j in inside:
            adjacency[i].append(j)
            adjacency[j].append(i)
    start = next(iter(inside))
    seen, stack = {start}, [start]
    while stack:
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == inside


def decode(spec, bits, encoding="binary"):
    """Return the island assignment a bitstring encodes, plus its checks."""
    buses, lines, k = spec["buses"], spec["lines"], spec["islands"]
    placement, unassigned = {}, []
    if encoding == "binary":
        for position, bus in enumerate(buses):
            placement[bus] = int(bits[position])
    else:
        for position, bus in enumerate(buses):
            chosen = [g for g in range(k) if bits[position * k + g] == 1]
            if len(chosen) != 1:
                unassigned.append(bus)
            if chosen:
                placement[bus] = chosen[0]

    islands = {g: sorted(b for b, i in placement.items() if i == g) for g in range(k)}
    cut = [(i, j) for i, j in lines if placement.get(i, -1) != placement.get(j, -2)]
    groups = spec["coherency"]
    coherent = all(len({placement.get(b, -1) for b in g}) == 1 for g in groups) and all(
        not ({placement.get(b, -1) for b in groups[a]} & {placement.get(b, -2) for b in groups[c]})
        for a in range(len(groups))
        for c in range(a + 1, len(groups))
    )
    checks = {
        "assigned": not unassigned,
        "coherency": coherent,
        "connected": all(connected(m, lines) for m in islands.values() if m),
        "generators": all(any(b in m for b in spec["generators"]) for m in islands.values()),
        "loads": all(any(b in m for b in spec["loads"]) for m in islands.values()),
        "min buses": all(len(m) >= 2 for m in islands.values()),
    }
    return {
        "islands": islands,
        "cut_lines": cut,
        "cut_weight": sum(line_weight(spec, i, j) for i, j in cut),
        "checks": checks,
        "feasible": all(checks.values()),
    }


references = {}
for name, spec in SYSTEMS.items():
    constant, linear, quadratic, variables = build_qubo(spec, "binary")
    energies = spectrum_of(constant, linear, quadratic, len(variables))
    order = np.argsort(energies, kind="stable")

    ground = decode(spec, to_bits(int(order[0]), len(variables)))
    best_feasible = next(
        (
            (int(index), decode(spec, to_bits(int(index), len(variables))))
            for index in order
            if decode(spec, to_bits(int(index), len(variables)))["feasible"]
        ),
        None,
    )
    references[name] = {
        "spec": spec, "energies": energies, "variables": variables,
        "qubo": (constant, linear, quadratic),
        "best_index": best_feasible[0], "best": best_feasible[1],
        "best_energy": float(energies[best_feasible[0]]),
    }

    failed = [check for check, ok in ground["checks"].items() if not ok]
    print(f"\n{name}")
    print(f"  QUBO ground state  energy {energies[order[0]]:9.4f}  "
          f"{'feasible' if not failed else 'FAILS: ' + ', '.join(failed)}")
    print(f"  best feasible      energy {references[name]['best_energy']:9.4f}  "
          f"cut weight {best_feasible[1]['cut_weight']:.4f}")
    print(f"  islands {best_feasible[1]['islands']}")
```

On the 9-bus system the QUBO's ground state is the answer. On the 14-bus system
it is not: the cheapest assignment the QUBO knows about leaves an island in two
disconnected pieces, and the best *usable* split costs noticeably more.

This is not a modeling mistake — it is a deliberate trade. Writing connectivity
as a quadratic penalty needs auxiliary variables and a great many terms, which
buys width and depth on a device that has neither to spare. Checking it after
decoding costs nothing. The consequence is that minimizing the QUBO is not the
same as solving the problem, and the gap has to be closed somewhere else. That
is the job of the postprocessing further down.

```python
def spring_layout(nodes, edges, seed=3, steps=400):
    """Lay out a graph with a force model: edges pull, every pair pushes."""
    rng = np.random.default_rng(seed)
    index = {node: position for position, node in enumerate(nodes)}
    position = rng.normal(0.0, 1.0, (len(nodes), 2))
    link = np.array([[index[i], index[j]] for i, j in edges])
    k = 1.0 / np.sqrt(len(nodes))
    for step in range(steps):
        delta = position[:, None, :] - position[None, :, :]
        distance = np.linalg.norm(delta, axis=-1) + 1e-9
        push = (delta / distance[..., None]) * (k**2 / distance)[..., None]
        push[np.arange(len(nodes)), np.arange(len(nodes))] = 0.0
        net = push.sum(axis=1)
        along = position[link[:, 0]] - position[link[:, 1]]
        length = np.linalg.norm(along, axis=-1, keepdims=True) + 1e-9
        pull = (along / length) * (length**2 / k)
        np.add.at(net, link[:, 0], -pull)
        np.add.at(net, link[:, 1], pull)
        limit = 0.1 * (1 - step / steps)
        norm = np.linalg.norm(net, axis=-1, keepdims=True) + 1e-9
        position += net / norm * np.minimum(norm, limit)
    position -= position.mean(axis=0)
    position /= np.abs(position).max()
    return {node: position[index[node]] for node in nodes}


def draw_split(axis, spec, solution, title, legend=False):
    """Draw a network with its islands colored and its cut lines dashed."""
    layout = spring_layout(spec["buses"], spec["lines"])
    cut = {frozenset(line) for line in solution["cut_lines"]}
    for i, j in spec["lines"]:
        style = (
            dict(color="#d62728", linestyle="--", linewidth=2.2)
            if frozenset((i, j)) in cut
            else dict(color="#c8c8c8", linewidth=1.2)
        )
        axis.plot(*zip(layout[i], layout[j]), zorder=1, **style)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for island, members in solution["islands"].items():
        if not members:
            continue
        points = np.array([layout[bus] for bus in members])
        axis.scatter(
            points[:, 0], points[:, 1], s=330, c=palette[island],
            edgecolors="white", linewidths=1.5, zorder=2, label=f"island {island}",
        )
    for bus in spec["buses"]:
        marker = "*" if bus in spec["generators"] else ""
        axis.annotate(
            f"{bus}{marker}", layout[bus], ha="center", va="center",
            color="white", fontsize=8, fontweight="bold", zorder=3,
        )
    axis.set_axis_off()
    axis.set_title(title, fontsize=10)
    if legend:
        axis.legend(loc="upper left", fontsize=8, framealpha=0.9)


figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
for position, (axis, name) in enumerate(zip(axes, SYSTEMS)):
    reference = references[name]
    draw_split(
        axis, reference["spec"], reference["best"],
        f"{name}: cut weight {reference['best']['cut_weight']:.3f}",
        legend=(position == 0),
    )
figure.suptitle("Best feasible split (generators marked *, cut lines dashed)")
figure.tight_layout()
```

## Running QAOA on the 14-bus model

The circuit construction is exactly the one from the QUBO tutorial: substitute
$x_i = (1 - z_i)/2$ to get an Ising Hamiltonian, then emit an `RZ` per field
term, a `CX`-`RZ`-`CX` ladder per coupling, and an `RX` mixer per layer.

```python
def to_ising(constant, linear, quadratic, n, tolerance=1e-9):
    """Rewrite a QUBO over spins, dropping coefficients that cancelled."""
    offset = constant + linear.sum() / 2 + sum(quadratic.values()) / 4
    field = -linear / 2
    for (i, j), value in quadratic.items():
        field[i] -= value / 4
        field[j] -= value / 4
    coupling = {key: value / 4 for key, value in quadratic.items()}
    field = np.where(np.abs(field) <= tolerance, 0.0, field)
    coupling = {key: value for key, value in coupling.items() if abs(value) > tolerance}
    return offset, field, coupling


NAME = "14-bus"
reference = references[NAME]
spec = reference["spec"]
constant, linear, quadratic = reference["qubo"]
N = len(reference["variables"])
spectrum = reference["energies"]
offset, field, coupling = to_ising(constant, linear, quadratic, N)
gamma_scale = max(abs(value) for value in coupling.values())

simulator = fq.simulator.Simulator(method="statevector")


def qaoa_program(betas, gammas, *, measure=False):
    program = fq.Program(N, N if measure else 0)
    for qubit in range(N):
        program.add(ops.H, qubit)
    for beta, gamma in zip(betas, gammas):
        for qubit, value in enumerate(field):
            if value != 0.0:
                program.add(ops.RZ(2.0 * gamma * value), qubit)
        for (i, j), value in coupling.items():
            program.add(ops.CX, (i, j))
            program.add(ops.RZ(2.0 * gamma * value), j)
            program.add(ops.CX, (i, j))
        for qubit in range(N):
            program.add(ops.RX(2.0 * beta), qubit)
    if measure:
        program.measure_all()
    return program


def probabilities(betas, gammas):
    result = simulator.run(
        qaoa_program(betas, gammas), shots=1, result_config={"final_state": True}
    ).result()
    return np.abs(result.get_statevector()) ** 2


print(f"{NAME}: {N} qubits, {len(coupling)} couplings, "
      f"{2 * len(coupling)} two-qubit gates per layer")
print(f"largest |J| = {gamma_scale:.3f}, Hamiltonian offset {offset:.3f}")
print(f"non-zero field terms: {int(np.count_nonzero(field))}")
```

The target to aim at is the best feasible split, not the QUBO ground state, so
that is what the probability below is measured against. Depth is grown one
layer at a time, seeding each depth from the previous solution.

```python
target_index = reference["best_index"]
target_energy = reference["best_energy"]
optimal_indices = np.flatnonzero(np.isclose(spectrum, spectrum[target_index]))
uniform_probability = optimal_indices.size / spectrum.size

rng = np.random.default_rng(2024)
depths = [1, 2, 3, 4]
mean_energies, hit_probabilities = [], []
betas = gammas = None

for depth in depths:
    def objective(parameters, depth=depth):
        return float(probabilities(parameters[:depth], parameters[depth:]) @ spectrum)

    if betas is None:
        starts = [
            np.concatenate(
                [rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / gamma_scale, depth)]
            )
            for _ in range(8)
        ]
    else:
        starts = [
            np.concatenate([betas, betas[-1:], gammas, gammas[-1:]]),
            np.concatenate([betas, [0.0], gammas, [0.0]]),
        ]

    best = min(
        (
            minimize(
                objective, start, method="COBYLA",
                options={"maxiter": 90, "rhobeg": np.pi / (4 * gamma_scale), "tol": 1e-8},
            )
            for start in starts
        ),
        key=lambda outcome: outcome.fun,
    )
    betas, gammas = best.x[:depth], best.x[depth:]

    distribution = probabilities(betas, gammas)
    mean_energies.append(float(best.fun))
    hit_probabilities.append(float(distribution[optimal_indices].sum()))
    print(
        f"p={depth}: mean energy {mean_energies[-1]:9.3f}   "
        f"P(best feasible) {hit_probabilities[-1]:.4f}   "
        f"{hit_probabilities[-1] / uniform_probability:5.1f}x uniform"
    )

print(f"best feasible energy {target_energy:.3f}, "
      f"QUBO ground energy {spectrum.min():.3f}")
```

```python
figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.2))

left.plot(depths, mean_energies, "o-", color="#1f77b4")
left.axhline(target_energy, color="#2ca02c", linestyle="--", label="best feasible")
left.axhline(spectrum.min(), color="#444444", linestyle="-.", label="QUBO ground state")
left.set_xlabel("QAOA depth $p$")
left.set_ylabel(r"$\langle H \rangle$")
left.set_title(f"{NAME}: mean energy")
left.set_xticks(depths)
left.legend(fontsize=8)

right.plot(depths, hit_probabilities, "s-", color="#d62728")
right.axhline(uniform_probability, color="#aaaaaa", linestyle=":", label="uniform sampling")
right.set_xlabel("QAOA depth $p$")
right.set_ylabel("probability of the best feasible split")
right.set_title("Solution probability")
right.set_xticks(depths)
right.legend(fontsize=8)

figure.tight_layout()
```

The mean energy falls toward the QUBO ground state, which sits *below* the best
feasible split — a reminder that the optimizer is minimizing the model, not the
problem. The probability of measuring the split we actually want rises with
depth, but it stays small. At this point a naive pipeline would need a very
large shot budget, and most of what it collected would be unusable.

## Postprocessing: pVSQA Method 1

The fix is to stop treating a measurement as an answer and start treating it as
a *starting point*. Shirai and Togawa's postprocessing variationally scheduled
quantum algorithm (pVSQA, IEEE Transactions on Quantum Engineering 5, 3100415,
2024) walks each measured bitstring downhill to the nearest feasible
assignment, so every shot yields a usable split.

Method 1 is the two-stage form. The first stage is the paper's Algorithm 1:
repeatedly flip whichever single variable most decreases

$$
Q' = Q_\text{objective} + A' \, Q_\text{violation},
$$

where the violation term counts constraint breaches *linearly* rather than
quadratically. That flatter penalty is the point — with $A'$ above the
objective's own scale, every local minimum of $Q'$ is feasible, so the descent
cannot stall on a violation.

Under the binary encoding there is no constraint left for the flip stage to
enforce, so it reduces to a pure descent on the QUBO. That is not the whole
job, as the next result shows.

```python
def greedy_repair(qubo, bits, constraints=(), multiplier=0.0, max_flips=500):
    """Flip the single most improving variable until none improves (Algorithm 1)."""
    constant, linear, quadratic = qubo
    values = np.asarray(bits, dtype=np.int8).copy()

    def score(candidate):
        total = constant + float(linear @ candidate)
        for (i, j), value in quadratic.items():
            total += value * candidate[i] * candidate[j]
        for members, lower, upper in constraints:
            reached = float(candidate[list(members)].sum())
            total += multiplier * max(0.0, reached - upper, lower - reached)
        return total

    current = score(values)
    for _ in range(max_flips):
        best_position, best_score = None, current
        for position in range(values.size):
            values[position] ^= 1
            candidate_score = score(values)
            values[position] ^= 1
            if candidate_score < best_score - 1e-12:
                best_position, best_score = position, candidate_score
        if best_position is None:
            break
        values[best_position] ^= 1
        current = best_score
    return values


rng = np.random.default_rng(7)
trials = [rng.integers(0, 2, N) for _ in range(300)]
stage_one = [greedy_repair(reference["qubo"], bits) for bits in trials]
feasible_after_stage_one = sum(decode(spec, bits)["feasible"] for bits in stage_one)
distinct = {"".join(map(str, bits)) for bits in stage_one}

print(f"stage 1 alone, from {len(trials)} random bitstrings:")
print(f"  {feasible_after_stage_one} feasible, {len(distinct)} distinct results")
failing = decode(spec, stage_one[0])["checks"]
print(f"  checks on a typical result: {failing}")
```

Every descent lands on the QUBO's ground state, and none of them is feasible —
because that ground state is the disconnected split found earlier. The greedy
stage is doing exactly what it was asked to do, and it is not enough.

The second stage handles what the QUBO deliberately left out. When an island
falls into several connected components, each stray fragment can be fixed two
ways: hand it to a neighboring island, or annex the buses along the shortest
path back to the island's main body. Exile is cheaper, but it is impossible
when a coherency group straddles the fragment. Both moves are constructed,
scored with the model's own energy, and the better one wins.

```python
def components_of(members, lines):
    """Return the connected components of the subgraph induced on a bus set."""
    inside = set(members)
    adjacency = {bus: [] for bus in inside}
    for i, j in lines:
        if i in inside and j in inside:
            adjacency[i].append(j)
            adjacency[j].append(i)
    seen, found = set(), []
    for start in sorted(inside):
        if start in seen:
            continue
        group, stack = [], [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        found.append(sorted(group))
    return found


def shortest_bridge(fragment, target, lines):
    """Return the buses strictly between two bus sets on a shortest path."""
    adjacency = {}
    for i, j in lines:
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    goal, source = set(target), set(fragment)
    previous = {bus: None for bus in fragment}
    queue = list(fragment)
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, ()):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            if neighbor in goal:
                path, walk = [], current
                while walk is not None and walk not in source:
                    path.append(walk)
                    walk = previous[walk]
                return sorted(path)
            queue.append(neighbor)
    return []


def energy_of(qubo, bits):
    constant, linear, quadratic = qubo
    total = constant + float(linear @ bits)
    for (i, j), value in quadratic.items():
        total += value * bits[i] * bits[j]
    return total


def repair_connectivity(spec, qubo, bits, max_moves=20):
    """Reconnect broken islands by exiling or annexing, whichever scores lower."""
    values = np.asarray(bits, dtype=np.int8).copy()
    positions = {bus: position for position, bus in enumerate(spec["buses"])}

    def moved(source, buses, island):
        candidate = source.copy()
        for bus in buses:
            candidate[positions[bus]] = island
        return candidate

    for _ in range(max_moves):
        islands = decode(spec, values)["islands"]
        placement = {bus: island for island, members in islands.items() for bus in members}
        change = None
        for island, members in islands.items():
            if len(members) < 2 or connected(members, spec["lines"]):
                continue
            pieces = components_of(members, spec["lines"])
            main = max(pieces, key=len)
            options = []
            for fragment in pieces:
                if fragment is main:
                    continue
                touching = {
                    placement[outside]
                    for i, j in spec["lines"]
                    for inside, outside in ((i, j), (j, i))
                    if inside in fragment and outside not in fragment
                    and placement.get(outside) not in (None, island)
                }
                for destination in touching or {1 - island}:
                    options.append(moved(values, fragment, destination))
                bridge = shortest_bridge(fragment, main, spec["lines"])
                if bridge:
                    options.append(moved(values, bridge, island))
            if options:
                change = min(options, key=lambda candidate: energy_of(qubo, candidate))
                break
        if change is None:
            break
        values = change
    return values


def repair_coherency(spec, bits):
    """Put each coherency group back on a single, distinct island."""
    values = np.asarray(bits, dtype=np.int8).copy()
    positions = {bus: position for position, bus in enumerate(spec["buses"])}
    taken = set()
    for group in spec["coherency"]:
        votes = {}
        for bus in group:
            island = int(values[positions[bus]])
            votes[island] = votes.get(island, 0) + 1
        order = sorted(range(spec["islands"]), key=lambda g: (-votes.get(g, 0), g))
        choice = next((g for g in order if g not in taken), order[0])
        taken.add(choice)
        for bus in group:
            values[positions[bus]] = choice
    return values


def pvsqa_method1(spec, qubo, bits, rounds=5):
    """Alternate the flip descent and the connectivity repair until both pass."""
    values = np.asarray(bits, dtype=np.int8).copy()
    best = None

    def consider(candidate):
        """Keep a candidate if it beats the best seen: feasibility, then energy."""
        nonlocal best
        solution = decode(spec, candidate)
        score = (solution["feasible"], -energy_of(qubo, candidate))
        if best is None or score > best[0]:
            best = (score, candidate.copy())
        return solution

    for _ in range(rounds):
        values = greedy_repair(qubo, values)
        values = repair_coherency(spec, values)
        if consider(values)["checks"]["connected"]:
            break
        values = repair_connectivity(spec, qubo, values)
        # The reconnected state is a candidate in its own right; the next
        # round's descent will usually abandon it for a lower, broken one.
        consider(values)
    return best[1]


repaired = [pvsqa_method1(spec, reference["qubo"], bits) for bits in trials]
feasible_repaired = [bits for bits in repaired if decode(spec, bits)["feasible"]]
optimal_repaired = [
    bits for bits in feasible_repaired
    if abs(energy_of(reference["qubo"], bits) - target_energy) < 1e-6
]

print(f"full Method 1, from the same {len(trials)} random bitstrings:")
print(f"  feasible: {len(feasible_repaired):3d} / {len(trials)} "
      f"({100 * len(feasible_repaired) / len(trials):.0f}%)")
print(f"  optimal:  {len(optimal_repaired):3d} / {len(trials)} "
      f"({100 * len(optimal_repaired) / len(trials):.0f}%)")
print(f"  best feasible energy reached {min(energy_of(reference['qubo'], b) for b in feasible_repaired):.3f}, "
      f"best possible {target_energy:.3f}")
print(f"  {len({''.join(map(str, bits)) for bits in repaired})} distinct results")
```

Together the two stages turn most random bitstrings into feasible splits, and
most of those are the best feasible split. Stage 2 is what made the difference:
the same descent that produced nothing usable on its own now lands on the
answer four times out of five.

That number is also a warning, and the next section takes it seriously.

## A control the demonstration needs

Repairing random bitstrings already reaches the best feasible split most of the
time. So before claiming that QAOA contributes anything, the honest thing to do
is run the classical half on its own and compare. Uniform random bitstrings,
repaired with the same Method 1, are the baseline; anything the quantum sampler
is worth has to show up as a difference against it.

```python
def optimal_rate(bitstrings):
    """Fraction of repaired samples that reach the best feasible split."""
    hits = 0
    for bits in bitstrings:
        fixed = pvsqa_method1(spec, reference["qubo"], bits)
        if (
            decode(spec, fixed)["feasible"]
            and abs(energy_of(reference["qubo"], fixed) - target_energy) < 1e-6
        ):
            hits += 1
    return hits / len(bitstrings)


BASELINE_SAMPLES = 300
rng = np.random.default_rng(11)
uniform_rate = optimal_rate([rng.integers(0, 2, N) for _ in range(BASELINE_SAMPLES)])

quantum_draw = simulator.run(
    qaoa_program(betas, gammas, measure=True),
    shots=BASELINE_SAMPLES,
    simulation_config={"seed": 17},
).result()
quantum_bitstrings = [
    np.array([int(character) for character in key], dtype=np.int8)
    for key, shots in quantum_draw.get_counts().items()
    for _ in range(shots)
]
quantum_rate = optimal_rate(quantum_bitstrings)

label = f"QAOA p={len(betas)} + repair"
print(f"reaching the best feasible split, {BASELINE_SAMPLES} samples each:")
print(f"  {'uniform random + repair':<24}: {uniform_rate:.3f}")
print(f"  {label:<24}: {quantum_rate:.3f}")
```

The two rates are close. On a 14-bus network the repair is strong enough to
find the answer from almost anywhere, so the quantum sampler has nothing left
to contribute, and no claim of advantage is supportable from this run. What
these results do establish is that the pipeline is correct end to end: the
model, the circuit, the repair, and the decoding all agree with an exhaustive
classical reference.

Separating the two claims needs a network where the classical descent gets
stuck — larger, more strongly meshed, or with tighter coherency groups. That
is exactly where the qubit accounting at the end of this tutorial becomes the
binding constraint, and it is the honest place to put the effort next.

## What postprocessing does to a QAOA run

Applied to a real measurement, the effect is a shift in what the shot budget
buys. Raw outcomes spread across the whole energy range and are mostly
infeasible; repaired outcomes collapse onto feasible splits. Repair runs once
per *distinct* outcome rather than once per shot, which is what keeps it
affordable.

```python
SHOTS = 2000
measured = simulator.run(
    qaoa_program(betas, gammas, measure=True),
    shots=SHOTS,
    simulation_config={"seed": 5},
).result()
counts = measured.get_counts()

raw_energies, raw_weights, fixed_energies, fixed_weights = [], [], [], []
feasible_shots_raw = feasible_shots_fixed = 0
best_solution, best_value = None, np.inf

for key, shots in counts.items():
    bits = np.array([int(character) for character in key], dtype=np.int8)
    raw_energies.append(energy_of(reference["qubo"], bits))
    raw_weights.append(shots)
    feasible_shots_raw += shots * decode(spec, bits)["feasible"]

    fixed = pvsqa_method1(spec, reference["qubo"], bits)
    value = energy_of(reference["qubo"], fixed)
    fixed_energies.append(value)
    fixed_weights.append(shots)
    solution = decode(spec, fixed)
    feasible_shots_fixed += shots * solution["feasible"]
    if solution["feasible"] and value < best_value:
        best_solution, best_value = solution, value

print(f"{len(counts)} distinct outcomes in {SHOTS} shots")
print(f"feasible shots before repair: {feasible_shots_raw:5d} / {SHOTS}")
print(f"feasible shots after repair:  {feasible_shots_fixed:5d} / {SHOTS}")
print(f"\nbest split found, energy {best_value:.3f} "
      f"(best possible {target_energy:.3f}):")
print(f"  islands    {best_solution['islands']}")
print(f"  cut lines  {[list(line) for line in best_solution['cut_lines']]}")
print(f"  cut weight {best_solution['cut_weight']:.4f}")
print(f"  checks     {best_solution['checks']}")
```

```python
figure, axis = plt.subplots(figsize=(7.5, 3.4))
bins = np.linspace(
    min(min(raw_energies), min(fixed_energies)),
    np.percentile(np.repeat(raw_energies, raw_weights), 99),
    45,
)
axis.hist(raw_energies, bins=bins, weights=raw_weights, color="#aab7c4",
          label="measured", edgecolor="white", linewidth=0.4)
axis.hist(fixed_energies, bins=bins, weights=fixed_weights, color="#d62728",
          label="after pVSQA Method 1", edgecolor="white", linewidth=0.4)
axis.axvline(target_energy, color="#2ca02c", linestyle="--", linewidth=2,
             label="best feasible split")
axis.set_xlabel("QUBO energy")
axis.set_ylabel("shots")
axis.set_title(f"{NAME}: {SHOTS} shots before and after repair")
axis.set_yscale("log")
axis.legend(fontsize=8)
figure.tight_layout()
```

## How wide does this get?

Qubit count is what decides whether a network is reachable at all, and it
follows directly from the encoding and the components in use. Three numbers
matter per system: the one-hot width $n_\text{buses} \times k$, the binary
width $n_\text{buses}$ when $k = 2$, and the width once the slack-carrying
requirements are added.

The slack-carrying components — "at least $M$ buses", "at least one generator",
"at least one load" — are inequalities, and an inequality needs a binary slack
register per island to absorb the difference. That is
$\lceil \log_2(\cdot) \rceil$ extra qubits per island, per requirement. The
balance requirement used above avoids all of it by being an equality instead,
which is why it is the default here.

```python
CATALOG = {
    #        buses  lines  islands  generators  loads
    "9-bus":  (9,     9,     2,       3,          4),
    "14-bus": (14,   20,     2,       5,         11),
    "24-bus": (24,   34,     3,      11,         13),
    "30-bus": (30,   41,     2,       6,         24),
    "39-bus": (39,   46,     3,      10,         29),
}

def slack_bits(maximum):
    return max(1, int(np.ceil(np.log2(max(1, maximum) + 1))))

rows = []
print(f"{'system':8} {'buses':>6} {'islands':>8} {'one-hot':>8} {'binary':>7} "
      f"{'+ slack':>8} {'narrowest':>10}")
for name, (buses, lines, islands, generators, loads) in CATALOG.items():
    one_hot = buses * islands
    binary = buses if islands == 2 else None
    slack = islands * (
        slack_bits(buses - 2) + slack_bits(generators - 1) + slack_bits(loads - 1)
    )
    narrowest = binary if binary is not None else one_hot
    rows.append((name, one_hot, binary, narrowest + slack))
    print(f"{name:8} {buses:6} {islands:8} {one_hot:8} "
          f"{'-' if binary is None else binary:>7} {narrowest + slack:8} "
          f"{narrowest:>10}")
```

```python
figure, axis = plt.subplots(figsize=(7.5, 3.4))
labels = [row[0] for row in rows]
positions = np.arange(len(rows))
width = 0.27

axis.bar(positions - width, [row[1] for row in rows], width,
         label="one-hot", color="#aab7c4")
axis.bar(positions, [row[2] if row[2] else 0 for row in rows], width,
         label="binary (two islands only)", color="#1f77b4")
axis.bar(positions + width, [row[3] for row in rows], width,
         label="narrowest + inequality slack", color="#d62728")
for position, row in zip(positions, rows):
    if row[2] is None:
        axis.annotate("n/a", (position, 1.5), ha="center", fontsize=7, color="#555555")
axis.set_xticks(positions)
axis.set_xticklabels(labels)
axis.set_ylabel("qubits")
axis.set_title("Encoding decides the width")
axis.legend(fontsize=8)
figure.tight_layout()
```

Two islands on the 30-bus system need 30 qubits with the binary encoding and 60
with one-hot — the difference between a statevector simulation that fits on a
laptop and one that does not. Three-island systems cannot use the binary
encoding at all, which is why the 24-bus and 39-bus rows show only the one-hot
width; a base-$k$ encoding recovers part of the saving there, at the cost of a
messier decoding.

## Where this leaves the problem

The pipeline is complete and each piece is checkable: the QUBO reproduces the
operating requirements, the two encodings agree on the answer, the circuit
construction matches an independent estimator, and every measured bitstring
becomes a feasible split.

What it does not yet show is a quantum advantage, and it is worth being precise
about why. On networks small enough to simulate, the classical repair reaches
the optimum from any starting point, so the quantum sampler's contribution is
unmeasurable. Making that contribution visible needs a network where the
classical descent has somewhere to get stuck — and there, the width numbers
above become the binding constraint rather than the runtime.

The parts most worth pushing on:

- **Narrower encodings.** A base-$k$ assignment beats one-hot for more than two
  islands, and merging buses that will never separate shrinks the problem
  before any encoding is chosen.
- **Constraint-preserving mixers.** An XY mixer confines the state to the
  one-hot subspace, so amplitude is never spent on assignments the penalty
  would only have to reject.
- **Warm starts.** Seeding QAOA from a classical partition and letting the
  circuit search nearby uses shallow depth where it helps most.
- **Noise.** Every run here is noiseless. Attaching a `fatqat.NoiseModel` to
  the simulator turns the same code into a study of how much depth this
  problem can actually afford on hardware.
