# fatqat.multisite — Multi-site Interaction Gates and Minimum-Width Layering

`fatqat.multisite` builds, decomposes, and parallelizes **multi-site interaction
gates**. It translates a restricted multi-body term (such as a PXP-model
three-body blockade term) into single- and two-qubit gates acting only on the
participating sites, and schedules a collection of such terms into the
**minimum number of parallel layers**.

---

## 1. Operator strings

A term is written as one character per site, each naming a single-site
operator:

| Char | Operator | Matrix (in `|0>, |1>` basis) | Role |
|------|----------|------------------------------|------|
| `P`  | `|0><0|` | `diag(1, 0)` | condition: site must be `0` |
| `Q`  | `|1><1|` | `diag(0, 1)` | condition: site must be `1` (`Q = I - P`) |
| `X`  | Pauli X | `[[0,1],[1,0]]` | rotation axis |
| `Y`  | Pauli Y | `[[0,-i],[i,0]]` | rotation axis |
| `Z`  | Pauli Z | `diag(1,-1)` | rotation axis |
| `I`  | identity | `diag(1,1)` | no-op (placeholder) |

Examples on sites `(n, m, k)`:

- `PXP` = `P ⊗ X ⊗ P` — rotate `m` about `X` when both neighbors are `|0>`.
- `QXQ` = `Q ⊗ X ⊗ Q` — rotate `m` when both neighbors are `|1>`.
- `PXQ` = `P ⊗ X ⊗ Q` — rotate `m` when left is `|0>` and right is `|1>`.
- `QXP` = `Q ⊗ X ⊗ P` — the mirrored condition.
- `XYZ`, `XX`, `PQP`, … — arbitrary mixes are supported.

Every string is a **multi-controlled rotation**: `P`/`Q` are the control
(condition) sites and `X`/`Y`/`Z` are the target (rotation) sites.

---

## 2. Decomposition

The decomposition is exact (up to a global phase) and uses only single-qubit
gates and CNOTs. It never needs ancilla qubits and never uses tensor-network
methods.

### 2.1 Z-string primitive

`z_string_rotation(sites, theta)` realizes

```
exp(-iθ · Z ⊗ Z ⊗ … ⊗ Z)  =  CNOT ladder + RZ(2θ) + reverse CNOT ladder
```

with `k-1` CNOTs on each side and one `RZ(2θ)` on the last site.

### 2.2 Diagonalize each single-site operator

`product_interaction` maps each label to an eigendecomposition `A = U D U†`:

```
P = (I+Z)/2,  Q = (I-Z)/2,  X = H Z H,  Y = (S H) Z (H Sdg),  Z = Z
```

The frame gates (`H`, `S`, `Sdg`) rotate the operator to `Z`; the diagonal
`D = diag(d0, d1)` is rewritten as `m·I + s·Z` with `m=(d0+d1)/2`,
`s=(d0-d1)/2`.

### 2.3 Z-string expansion

```
⊗_j (m_j·I + s_j·Z_j) = Σ_{S ⊆ sites} c_S · (⊗_{j∈S} Z_j)
```

All Z-strings commute (they are diagonal), so

```
exp(-iθ · ⊗A_j) = (⊗U) · [ Π_S exp(-iθ·c_S · ⊗_{j∈S} Z_j) ] · (⊗U†)
```

holds exactly. Each factor is the 2.1 primitive. The cost is `2^k` terms for
`k` sites, which is the reason the target range is `k = 3…5`.

---

## 3. Minimum-width layering

`layer_schedule(terms, optimal=True)` partitions site tuples into layers such
that terms in the same layer share no site. Two terms that share a site must
be in different layers, so the minimum number of layers equals the chromatic
number of the **conflict graph** (vertices = terms, edges = shared sites).

The solver is exact branch-and-bound:

1. **Clique lower bound** — the largest set of terms containing any one site
   (for a PXP chain this is 3, so 3 layers is a hard lower bound).
2. **DSATUR greedy** — a valid coloring as the initial upper bound.
3. **Backtracking search** — assign colors in saturation order, prune a branch
   as soon as its color count reaches the current best.

For a PXP chain the lower and upper bounds meet at 3, so the result is returned
immediately. `optimal=False` uses the linear-time greedy instead.

---

## 4. API

Decomposition functions return `list[(Operation, tuple[int, ...])]` —
`(gate, target_qubits)` pairs — so they can be inspected, rewritten, or merged
before being turned into a program.

| Function | Purpose |
|----------|---------|
| `interaction(sites, opstring, theta)` | main entry point: operator string → instructions |
| `controlled_rotation(controls, control_states, target, theta, axis="X")` | explicit multi-controlled rotation with per-control states |
| `product_interaction(operators, theta)` | general product of single-site operators `[(site, label), …]` |
| `pauli_string_rotation(sites, paulis, theta)` | pure Pauli-string rotation |
| `z_string_rotation(sites, theta)` | Z-string rotation primitive |
| `layer_schedule(terms, optimal=True)` | minimum-width parallel layering |
| `build_layered_programs(terms, theta, opstring=…, num_qubits=…)` | decompose + layer + one `Program` per layer |
| `to_program(instructions, num_qubits)` | instructions → `fatqat.Program` |
| `append_instructions(program, instructions)` | append instructions to an existing `Program` |

### Example

```python
import fatqat.multisite as ms

# Decompose PXP on sites (0, 1, 2).
instr = ms.interaction((0, 1, 2), "PXP", 0.4)

# Equivalent explicit form: P X Q.
instr2 = ms.controlled_rotation((0, 2), (0, 1), 1, 0.4)

# Minimum-layer schedule of an open PXP chain.
chain = [(i, i + 1, i + 2) for i in range(6)]
layers = ms.layer_schedule(chain, optimal=True)      # 3 layers

# One Program per layer.
programs = ms.build_layered_programs(chain, theta=0.3, opstring="PXP", num_qubits=8)
```

---

## 5. Application scenarios

- **PXP / Rydberg-blockade dynamics.** Trotterize `H_PXP = Σ Ω·P_{i-1} X_i P_{i+1}`:
  each term is the string `PXP`, and `build_layered_programs` compresses one
  Trotter step into three parallel layers.
- **Constrained (blockade / anti-blockade) models.** `QXQ`, `PXQ`, `QXP` encode
  directional or excited-state constraints; the string alphabet expresses any
  projector/Pauli product on 3–5 sites.
- **Hamiltonian simulation of k-local terms.** Any product of single-site Pauli
  and projector operators (`XX`, `XYZ`, `PZQ`, …) becomes a 1- and 2-qubit
  circuit, feeding directly into Trotter product formulas.
- **Multi-controlled rotations in algorithms.** A `P`/`Q`-controlled
  `X`/`Y`/`Z` rotation is a Toffoli-style conditional gate for oracles, Grover,
  and phase-estimation subroutines.
- **Depth minimization.** `layer_schedule` yields the minimum-depth schedule for
  commuting disjoint terms, directly reusable as a circuit-scheduling step.

---

## 6. fatqat functions used

**In the module (`fatqat/multisite.py`):**

- `fatqat.operations.Operation` — base class used for the `Instruction` type.
- `fatqat.operations.CX` — the two-qubit CNOT gate (the only two-qubit gate emitted).
- `fatqat.operations.H` — Hadamard frame gate.
- `fatqat.operations.S`, `fatqat.operations.Sdg` — phase frame gates for the `Y` axis.
- `fatqat.operations.RZ(theta)` — the single-qubit rotation carrying the angle.
- `fatqat.Program(n)` and `Program.add(op, targets)` — program construction.

**In tests / examples:**

- `fatqat.simulator.Simulator(method="unitary", runtime="numpy")` — compute the full unitary.
- `Simulator.run(program, result_config={"counts": False, "final_state": True})` — execute.
- `Result.get_unitary()` — retrieve the unitary for verification.
- (`Simulator(method="statevector", runtime="numpy")` works identically for state evolution.)

---

## 7. Required Python packages

- **Python ≥ 3.12** — fatqat's minimum version.
- **fatqat** — the package this module extends.
- **numpy** — fatqat's numerical core (already a fatqat dependency).
- **scipy** — used for the exact reference unitaries (`scipy.linalg.expm`) in the
  test suite; not required to use the module itself.
- **pytest** — to run the test suite.

---

## 8. Verification

- 55 pytest tests pass: each decomposition's unitary is compared against
  `scipy.linalg.expm` of the exact target operator, covering `PXP`/`QXQ`/`PXQ`/
  `QXP`/`XYZ`, 3/4/5-body terms, mixed control states, and end-to-end layering.
  The minimum layer count is cross-checked against brute-force enumeration.
- 17 doctests pass; pylint 10.00/10; black clean.
