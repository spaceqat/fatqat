---
title: "Search quantum circuits for ADMET prediction"
description: "Reimplement the QCS-ADME search, imbalance-aware scoring, regression scoring, and training workflow with FatQat parameter sweeps."
icon: material-flask-outline
figure_alts:
  - "Classification and regression search scores for four candidate quantum circuits on synthetic inputs."
  - "Three-qubit circuit selected by the classification search."
  - "Training mean squared error for the selected classification and regression circuits."
  - "Held-out classification confusion matrix and regression predictions on synthetic inputs."
---

# Search quantum circuits for ADMET prediction

## What ADMET prediction measures

A molecule's activity against a biological target is only part of its
potential as a drug. **ADMET** describes how the body handles a compound
and whether it may cause harm:

| Property | What it describes |
| --- | --- |
| **Absorption** | Entry from the administration site into the bloodstream. |
| **Distribution** | Movement from blood into tissues. |
| **Metabolism** | Chemical transformation of the compound in the body. |
| **Excretion** | Removal of the compound and its metabolites. |
| **Toxicity** | Potential harmful effects, such as liver injury. |

Predicting these properties from molecular structure helps researchers
prioritize candidates before costly experiments. These endpoints require different learning objectives. Blood–brain barrier
penetration and drug-induced liver injury are classification examples;
drug half-life and clearance require continuous predictions. Unequal class
counts can hide poor minority-class performance. Continuous targets instead
require preserving degrees of similarity. These two challenges motivate
the task-specific circuit scores used below.

## From molecular features to quantum circuit search

Which circuit should we train when its inputs describe a molecule? The
QCS-ADME workflow searches circuit structures before optimizing their
weights. It compares the states produced by different inputs with a desired
similarity matrix, then combines that score with resistance to simulated
noise. Classification asks for similar states within a class; regression
asks for similar states when target values are close.

This tutorial reimplements that loop with [`Program`][fatqat.Program],
[`ParameterVector`][fatqat.ParameterVector], and
[`Estimator`][fatqat.Estimator]. You will generate a small candidate pool,
score both tasks, train one circuit per task, and evaluate predictions on
held-out inputs. Each circuit is built once; features and weights enter as
parameter bindings throughout search and training.

The scientific reference is [*QCS-ADME: Quantum Circuit Search for Drug Property Prediction with
Imbalanced Data and Regression Adaptation*](https://arxiv.org/abs/2503.01927)
(2026). The paper provides the motivation and search method.
The executable formulas below follow the supplied Quantum ADMET reference
implementation, including its binary imbalance weighting, one-qubit
reduced-state score, regression target weighting, composite ranking, and
mean squared training loss. Its binary weighting and reduced-state similarity
use implementation-specific conventions, which are specified below.

To keep execution small and offline, we use three qubits, four candidates,
synthetic inputs, a fixed line of permitted CX edges, and illustrative
depolarizing noise. These replace the original study's large candidate pool,
ADMET datasets, and device-calibration-driven generation. COBYLA replaces
TorchQuantum autodifferentiation and Adam. The results demonstrate the
workflow; they are not ADMET benchmark results or a reproduction of the
original experiment. The final section shows how to load the project's
existing feature arrays without importing TorchQuantum or PennyLane.

## Separate training inputs from evaluation inputs

Real ADMET inputs can be prepared molecular descriptors or fingerprints.
Here, six synthetic angles stand in for those features. A nonlinear rule
creates an imbalanced binary task and a separate continuous target. We
generate the training and test examples independently. Search and training
use only the training split; the test split is evaluated after both models
have been fixed.

Binary labels use the implementation's convention, $-1$ and $+1$. Regression
targets lie in $[-1,1]$ because our readout is a single Pauli-Z expectation.
No transformation is fitted on the test split.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import fatqat as fq
import fatqat.operations as ops

NUM_QUBITS = 3
NUM_FEATURES = 6
NUM_WEIGHTS = 6
NUM_CANDIDATES = 4
NUM_PARAMETER_SAMPLES = 4
NUM_DECOYS = 4
NOISE_IMPORTANCE = 0.25
SIGMA = 1.0


def synthetic_split(count, seed):
    """Generate illustrative angles and two targets, without molecular data."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, np.pi, (count, NUM_FEATURES))
    signal = 0.65 * np.cos(x[:, 0]) + 0.35 * np.sin(x[:, 1])
    binary = np.where(signal > 0.55, 1, -1)
    continuous = 0.7 * np.cos(x[:, 0]) - 0.3 * np.sin(x[:, 2])
    return x, binary, continuous


x_train, y_class_train, y_reg_train = synthetic_split(48, 10)
x_test, y_class_test, y_reg_test = synthetic_split(24, 11)
tasks = {
    "classification": (x_train, y_class_train, x_test, y_class_test),
    "regression": (x_train, y_reg_train, x_test, y_reg_test),
}
print("Synthetic training class counts:", np.unique(y_class_train, return_counts=True))
print(f"{len(x_train)} training inputs; {len(x_test)} held-out inputs per task")
```

## Generate candidates as parameterized programs

Each candidate consumes every feature once and has the same weight budget.
The seed controls feature order, rotation axes, and CX placements. We
restrict the two-qubit gates to a bidirectional three-qubit line. An initial
Hadamard layer makes an initial RZ encoding observable; subsequent rotations
and entanglers mix feature and weight effects.

The short list of gate specifications also lets us build Clifford decoys
with the same gate locations. It belongs to this tutorial, not to FatQat's
public API. For device-aware studies, replace the generator with candidates
that respect your selected device's gate set, connectivity, and calibration.

```python
FEATURES = fq.ParameterVector("x", NUM_FEATURES)
WEIGHTS = fq.ParameterVector("theta", NUM_WEIGHTS)
EDGES = [(q, q + 1) for q in range(NUM_QUBITS - 1)]
EDGES += [(target, control) for control, target in EDGES]
ROTATIONS = (ops.RX, ops.RY, ops.RZ)
CLIFFORDS = (ops.H, ops.S, ops.Z, ops.X, ops.Y)


def generate_candidate(seed):
    """Return a program and gate locations for a seeded candidate."""
    rng = np.random.default_rng(seed)
    program = fq.Program(NUM_QUBITS)
    locations = []

    def add(gate, wires):
        program.add(gate, wires)
        locations.append(wires)

    for q in range(NUM_QUBITS):
        add(ops.H, (q,))
    feature_order = rng.permutation(NUM_FEATURES)
    weight_index = 0
    for position, feature in enumerate(feature_order):
        q = position % NUM_QUBITS
        rotation = ROTATIONS[int(rng.integers(len(ROTATIONS)))]
        add(rotation(FEATURES[int(feature)]), (q,))
        # Spread the weight budget across the encoding gates.
        while weight_index < (position + 1) * NUM_WEIGHTS // NUM_FEATURES:
            rotation = ROTATIONS[int(rng.integers(len(ROTATIONS)))]
            add(rotation(WEIGHTS[weight_index]), (q,))
            weight_index += 1
        if position % NUM_QUBITS == NUM_QUBITS - 1:
            for _ in range(int(rng.integers(1, NUM_QUBITS + 1))):
                add(ops.CX, EDGES[int(rng.integers(len(EDGES)))])
    return program, locations


candidates = [generate_candidate(100 + i) for i in range(NUM_CANDIDATES)]
ideal_backend = fq.simulator.Simulator(method="statevector", runtime="numpy")
estimator = fq.Estimator(ideal_backend)
BLOCH = [
    fq.Observable.from_sparse([(axis, (0,), 1.0)], num_qubits=NUM_QUBITS)
    for axis in ("X", "Y", "Z")
]


def expectations(program, weights, features, observables, runner=estimator):
    """Evaluate one shared weight vector over a batch of feature rows."""
    bound = program.assign_parameters({WEIGHTS: weights})
    results = runner.run_sweep(
        bound, observables, {FEATURES: features}, shots=0
    ).result()
    return np.asarray([result.get_expectation() for result in results])
```

`shots=0` gives exact expectations. The sweep returns one result per input
row; it does not require a new circuit construction for each example. The
three observables expose the reduced state of qubit 0 through its Bloch
vector $r_i=(\langle X_0\rangle,\langle Y_0\rangle,\langle Z_0\rangle)$.

## Reproduce the task-specific representational scores

For one measured qubit,

$$
\rho_i=\frac{I+r_i\cdot\sigma}{2},\qquad
\operatorname{Tr}(\rho_i\rho_j)=\frac{1+r_i\cdot r_j}{2}.
$$

The reference implementation squares this overlap for off-diagonal pairs
and sets the diagonal to one. We preserve that convention:
$S_{ij}=[(1+r_i\cdot r_j)/2]^2$ for $i\ne j$, and $S_{ii}=1$.
This is the project's similarity heuristic, not mixed-state quantum fidelity.
The forced diagonal also differs from squared purity for a mixed reduced
state. Using the Bloch vectors avoids constructing and partially tracing a
full density matrix for every input pair.

For classification, $A_{ij}=1$ when labels match and zero otherwise. Class
weights $w_c=N/(2n_c)$ come from the **entire training split**, even though
we score a balanced subset. The implementation upweights cross-class pairs by
the minority weight $w_m$ and leaves same-class pairs at one. Call this
importance matrix $W$. For each sampled weight vector we threshold $S$ at
its $W$-weighted mean, including the diagonal, then average the resulting
binary matrices to obtain $\bar B$:

$$
R_{\rm cls}=1-\frac{\sum_{ij}[W_{ij}(A_{ij}-\bar B_{ij})]^2}
                         {2\sum_{ij}A_{ij}}.
$$

The weight is squared along with the error, so this score can be negative
under severe imbalance. We do not clip it or interpret it as accuracy.

For regression, the desired similarity is
$A_{ij}=\exp[-(y_i-y_j)^2/(2\sigma^2)]$. The implementation additionally weights
the circuit similarity by $G_{ij}=\exp[-(y_i-y_j)^2/2]$, with fixed bandwidth
one, before computing $R_{\rm reg}=1-\operatorname{mean}[(A-G\bar S)^2]$.
Both uses of the labels are intentional here. This is a supervised search
score computed on training examples, not a predictive kernel to evaluate
using unknown test labels. Regression target scaling changes both Gaussian
weights and therefore must be recorded when comparing experiments.

```python
def reduced_similarity(bloch):
    """The reference's one-qubit squared-overlap convention."""
    overlap = np.clip((1.0 + bloch @ bloch.T) / 2.0, 0.0, 1.0)
    similarity = overlap**2
    np.fill_diagonal(similarity, 1.0)
    return similarity


def score_candidate(program, features, labels, task, parameter_samples, class_weight):
    """Compute the reference classification or regression search score."""
    similarities = np.asarray(
        [
            reduced_similarity(expectations(program, weights, features, BLOCH))
            for weights in parameter_samples
        ]
    )
    if task == "classification":
        ideal = (labels[:, None] == labels[None, :]).astype(float)
        importance = np.where(ideal == 1, 1.0, class_weight)
        thresholds = np.average(
            similarities.reshape(len(parameter_samples), -1),
            axis=1,
            weights=importance.ravel(),
        )
        mean_thresholded = (similarities > thresholds[:, None, None]).mean(axis=0)
        return 1.0 - np.sum((importance * (ideal - mean_thresholded)) ** 2) / (
            2.0 * ideal.sum()
        )
    differences = labels[:, None] - labels[None, :]
    ideal = np.exp(-(differences**2) / (2.0 * SIGMA**2))
    target_weights = np.exp(-(differences**2) / 2.0)
    return 1.0 - np.mean((ideal - target_weights * similarities.mean(axis=0)) ** 2)


search_rng = np.random.default_rng(20)
parameter_samples = search_rng.uniform(
    0.0, 2.0 * np.pi, (NUM_PARAMETER_SAMPLES, NUM_WEIGHTS)
)
representation_scores = {}
for task, (train_x, train_y, _, _) in tasks.items():
    if task == "classification":
        labels, counts = np.unique(train_y, return_counts=True)
        per_class = min(4, int(counts.min()))
        selected = np.concatenate(
            [
                search_rng.choice(
                    np.flatnonzero(train_y == label), per_class, replace=False
                )
                for label in labels
            ]
        )
        minority_weight = len(train_y) / (2.0 * counts.min())
    else:
        selected = search_rng.choice(len(train_y), min(8, len(train_y)), replace=False)
        minority_weight = 1.0
    representation_scores[task] = np.asarray(
        [
            score_candidate(
                program,
                train_x[selected],
                train_y[selected],
                task,
                parameter_samples,
                minority_weight,
            )
            for program, _ in candidates
        ]
    )
```

Every candidate sees the same examples and sampled weight vectors. This
reduces incidental variation between candidates. Four weight samples are a
small execution budget, not evidence that the ranking has converged. A
research run should examine ranking stability across sample counts and seeds.

## Add Clifford noise resilience and choose candidates

For each candidate we replace every one-qubit gate by a random choice from
H, S, Z, X, and Y, and retain the CX locations. These are the implementation's
Clifford decoys. We compare their ideal and noisy computational-basis
probability distributions using total variation distance:

$$
\mathrm{CNR}=1-\frac{1}{D}\sum_{d=1}^{D}
\frac12\sum_z|p_d(z)-\tilde p_d(z)|,\qquad
\mathrm{score}=R\,\mathrm{CNR}^{\alpha}.
$$

Here $\alpha=0.25$. The illustrative noise model applies depolarization
after each CX. Exact density-matrix probabilities remove shot fluctuations
from this small comparison. The CNR value describes these decoys under
this noise model; it does not estimate the fidelity of an IBM device.

```python
noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.03), operation=ops.CX)
noisy_backend = fq.simulator.Simulator(
    method="density_matrix", runtime="numpy", noise=noise
)


def clifford_resilience(locations, seed):
    """Compare exact decoy distributions under ideal and noisy simulation."""
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(NUM_DECOYS):
        decoy = fq.Program(NUM_QUBITS)
        for wires in locations:
            gate = ops.CX if len(wires) == 2 else CLIFFORDS[int(rng.integers(5))]
            decoy.add(gate, wires)
        state = ideal_backend.run(decoy, shots=0).result().get_statevector()
        density = noisy_backend.run(decoy, shots=0).result().get_density_matrix()
        ideal_probabilities = np.abs(state) ** 2
        noisy_probabilities = np.diag(density).real
        distances.append(0.5 * np.abs(ideal_probabilities - noisy_probabilities).sum())
    return 1.0 - np.mean(distances)


cnr = np.asarray(
    [
        clifford_resilience(locations, 200 + i)
        for i, (_, locations) in enumerate(candidates)
    ]
)
composite_scores = {
    task: scores * np.clip(cnr, 1e-10, None) ** NOISE_IMPORTANCE
    for task, scores in representation_scores.items()
}
selected_candidates = {
    task: int(scores.argmax()) for task, scores in composite_scores.items()
}
for task, scores in composite_scores.items():
    print(task, "representation:", np.round(representation_scores[task], 4))
    print(
        task, "composite:", np.round(scores, 4), "selected:", selected_candidates[task]
    )
print("CNR:", np.round(cnr, 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (task, scores) in zip(axes, composite_scores.items()):
    positions = np.arange(NUM_CANDIDATES)
    ax.bar(
        positions - 0.18, representation_scores[task], width=0.36, label="Task score"
    )
    ax.bar(positions + 0.18, scores, width=0.36, label="With CNR")
    ax.set(
        title=task.capitalize(),
        xlabel="Candidate",
        ylabel="Search score",
        xticks=positions,
    )
    ax.legend()
fig.suptitle("Circuit search on synthetic inputs")
fig.tight_layout()
```

The highest composite score determines which circuit we train for each
task. These two tasks need not select the same circuit. As in the
implementation, the multiplicative rule has a limitation: when $R$ is negative, reducing
CNR makes the product less negative and can improve its rank. Inspect the
uncombined scores when using strongly imbalanced data; do not treat this
formula as a universally valid noise penalty.

```python
selected_program = candidates[selected_candidates["classification"]][0]
figure = selected_program.draw("matplotlib")
figure.set_size_inches(14, 4)
```

## Train the selected circuits

The prediction is $f_\theta(x)=\langle Z_0\rangle$. We minimize
$N^{-1}\sum_i(f_\theta(x_i)-y_i)^2$ for both tasks, preserving the implementation's
single-output MSE objective. Imbalance weighting was used in architecture
selection; this loss itself is unweighted. A binary prediction is $+1$ for
a positive expectation and $-1$ otherwise. Regression uses the expectation
directly, so its output is bounded to $[-1,1]$.

COBYLA calls the exact FatQat estimator as a black-box function. The small
evaluation budget keeps the tutorial runnable; reaching that budget is not
a claim of convergence. We print the optimizer status alongside the loss.

```python
trained = {}
traces = {}
for task, (train_x, train_y, _, _) in tasks.items():
    program = candidates[selected_candidates[task]][0]
    trace = []

    def objective(weights):
        predictions = expectations(program, weights, train_x, BLOCH[2])
        loss = float(np.mean((predictions - train_y) ** 2))
        trace.append(loss)
        return loss

    initial = np.random.default_rng(30).uniform(-0.2, 0.2, NUM_WEIGHTS)
    fit = minimize(
        objective,
        initial,
        method="COBYLA",
        options={"maxiter": 60, "rhobeg": 0.5},
    )
    trained[task] = (program, fit.x)
    traces[task] = trace
    print(f"{task}: MSE {trace[0]:.4f} -> {fit.fun:.4f}; {fit.nfev} evaluations")
    print("Optimizer:", fit.message)

fig, ax = plt.subplots(figsize=(7, 4))
for task, trace in traces.items():
    ax.plot(trace, label=task)
ax.set(
    xlabel="Loss evaluation", ylabel="Training MSE", title="Selected-circuit training"
)
ax.legend()
fig.tight_layout()
```

## Evaluate held-out predictions

For classification, compare accuracy with an always-majority prediction
and report balanced accuracy, the mean recall of the two classes. A high
ordinary accuracy alone can conceal failure on the minority class. For
regression, compare mean absolute error with a constant prediction equal
to the training median. These are simple checks on whether the fitted
circuit learned useful structure in this synthetic exercise.

```python
test_predictions = {}
for task, (_, _, test_x, _) in tasks.items():
    program, weights = trained[task]
    test_predictions[task] = expectations(program, weights, test_x, BLOCH[2])

_, train_labels, _, test_labels = tasks["classification"]
predicted_labels = np.where(test_predictions["classification"] > 0, 1, -1)
label_order = np.array([-1, 1])
confusion = np.asarray(
    [
        [
            np.sum((test_labels == actual) & (predicted_labels == predicted))
            for predicted in label_order
        ]
        for actual in label_order
    ]
)
if np.any(confusion.sum(axis=1) == 0):
    raise ValueError(
        "Both classes are needed in the held-out split for balanced accuracy."
    )
accuracy = np.mean(predicted_labels == test_labels)
balanced_accuracy = np.mean(np.diag(confusion) / confusion.sum(axis=1))
majority = label_order[
    np.argmax([np.sum(train_labels == label) for label in label_order])
]
print(
    f"Classification accuracy: {accuracy:.3f}; majority baseline: {np.mean(test_labels == majority):.3f}"
)
print(f"Balanced accuracy: {balanced_accuracy:.3f}; constant-class baseline: 0.500")

_, train_targets, _, test_targets = tasks["regression"]
reg_predictions = test_predictions["regression"]
mae = np.mean(np.abs(reg_predictions - test_targets))
baseline_mae = np.mean(np.abs(np.median(train_targets) - test_targets))
print(f"Regression MAE: {mae:.3f}; training-median baseline MAE: {baseline_mae:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(confusion, cmap="Blues")
for row in range(2):
    for column in range(2):
        axes[0].text(
            column,
            row,
            str(confusion[row, column]),
            ha="center",
            va="center",
            color="white" if confusion[row, column] > confusion.max() / 2 else "black",
        )
axes[0].set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=label_order,
    yticklabels=label_order,
    xlabel="Predicted label",
    ylabel="True label",
    title="Classification",
)
axes[1].scatter(test_targets, reg_predictions, alpha=0.8)
axes[1].plot([-1, 1], [-1, 1], "k--", label="Perfect prediction")
axes[1].set(
    xlabel="True target",
    ylabel="Predicted expectation",
    title="Regression",
    xlim=(-1.05, 1.05),
    ylim=(-1.05, 1.05),
)
axes[1].legend()
fig.suptitle("Held-out synthetic examples")
fig.tight_layout()
```

Search scores and held-out predictive performance measure different things.
The score selects an architecture using a small set of untrained states;
optimization, data coverage, and readout constraints still determine the
trained model's performance. Poor results are informative and should not be
hidden by tuning on the test examples.

With the seeds above, classification reaches about 0.58 balanced accuracy,
while regression MAE is about 0.69, worse than the constant baseline's 0.48.
In this run the selected regression architecture's high search score does
not translate into useful predictions. This is a reason to validate the
search heuristic on a validation split, rather than assume its ranking
predicts generalization.

## Use the existing Quantum ADMET datasets

The original loader reads `x_train.npy`, `y_train.npy`, `x_test.npy`, and
`y_test.npy` from `Elivagar/experiment_data/<dataset>/`, keeps the first
128 input columns, and passes them directly to rotation gates for `angle`
encoding. In particular, it does **not** multiply those values by $\pi$.
The following loader retains that behavior for one data repetition. It
requires finite numeric arrays, matching row counts, binary labels already
encoded as $\{-1,+1\}$, or regression targets already scaled to $[-1,1]$.
It fails on incompatible data instead of silently changing the experiment.

```python
def load_admet_split(directory, task, num_features=128):
    """Load one prepared local dataset in the Quantum ADMET array format."""
    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    if not isinstance(num_features, int) or num_features < 1:
        raise ValueError("num_features must be a positive integer")
    arrays = []
    for split in ("train", "test"):
        x = np.load(Path(directory) / f"x_{split}.npy", allow_pickle=False)
        y = np.load(Path(directory) / f"y_{split}.npy", allow_pickle=False)
        if x.ndim != 2 or x.shape[1] < num_features:
            raise ValueError(
                f"{split} features must have at least {num_features} columns"
            )
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1 or len(y) != len(x) or len(y) == 0:
            raise ValueError(
                f"{split} labels must have one value per nonempty feature row"
            )
        if any(a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in (x, y)):
            raise ValueError(f"{split} arrays must contain finite real numbers")
        if task == "classification" and set(np.unique(y)) != {-1, 1}:
            raise ValueError(f"{split} labels must contain both -1 and +1")
        if task == "regression" and np.any(np.abs(y) > 1):
            raise ValueError(f"{split} regression targets must already lie in [-1, 1]")
        arrays.extend((x[:, :num_features].astype(float), y.astype(float)))
    return tuple(arrays)
```

In a downloaded notebook or script, move this loader above the data setup,
replace the synthetic `tasks` assignment with the following, and set
`NUM_QUBITS = 7`, `NUM_FEATURES = 128`, and `NUM_WEIGHTS = 32` **before**
constructing the parameter vectors and candidate programs. Set `root` to
your local checkout. These lines are not executed in the documentation
build, which always uses the synthetic example.

~~~python
root = Path("/path/to/quantum_admet/Elivagar/experiment_data")
tasks = {
    "classification": load_admet_split(root / "pampa", "classification"),
    "regression": load_admet_split(root / "clearance_hepatocyte_az", "regression"),
}
~~~

Keep the prepared split's provenance and feature-generation procedure with
your experiment. This loader does not establish whether upstream descriptor
selection or target scaling used only training data. If you regenerate
features, fit preprocessing on training molecules and apply the same fitted
transformation to validation and test molecules. Preserve the intended
molecular split rather than replacing it with a random row split.

For a larger study, increase the candidate and sampling budgets, select
hyperparameters on a separate validation split, and evaluate the test split
once. Preserve candidates, seeds, optimized weights, target transforms, and
noise settings with the results. Existing TorchQuantum checkpoints and
device mappings are not consumed by this tutorial: reusing a particular
legacy circuit requires translating its ordered gates and parameter bounds
into a `Program` and verifying its outputs before retraining.

Next, use [ideal and noisy simulation](../guide/ideal-and-noisy.md) to
evaluate the fitted circuit under the same noise model used for selection,
or [the Estimator reference](../api/estimator.md) to introduce sampled
expectations. When sampling sweeps, reusing a seed can correlate errors
between rows; exact expectations here avoid that issue.
