# Gates

All gates live in the `fqc.ops` namespace. The full list with signatures is
in the [API reference](../api/operations.rst); this page covers how to use
them.

```{currentmodule} fatqcat.operations
```

## Singletons vs. classes

Fixed (parameter-free) gates are exported as ready-to-use singleton values:

```python
program.add(fqc.ops.H, 0)
program.add(fqc.ops.X, 0)
```

Parametric gates are exported as classes and must be instantiated with their
parameter before use:

```python
program.add(fqc.ops.RX(0.2), 0)
program.add(fqc.ops.CPhase(1.5), (0, 1))
```

Passing an unparameterized class where a value is expected (e.g.
`fqc.ops.RX` instead of `fqc.ops.RX(0.2)`) raises a {py:exc}`TypeError` from
{py:meth}`~fatqcat.Program.add` that names the mistake.

## Gate categories

- **Single-qubit fixed**: {py:data}`I`, {py:data}`H`, {py:data}`S`,
  {py:data}`Sdg`, {py:data}`T`, {py:data}`Tdg`, {py:data}`X`, {py:data}`Y`,
  {py:data}`Z`.
- **Parametric** (instantiate with an angle in radians): {py:class}`RX`,
  {py:class}`RY`, {py:class}`RZ`, {py:class}`Phase`, and the two-qubit
  {py:class}`CPhase`.
- **Multi-qubit fixed**: {py:data}`CX`, {py:data}`CZ`, {py:data}`Swap`,
  {py:data}`CY`, {py:data}`CS`, {py:data}`iSwap`, {py:data}`CCX`,
  {py:data}`CSwap`. For controlled gates, operand order is
  `(control, ..., target)`.
- **Dimension-generic (qudit)**: {py:class}`Shift` and {py:class}`Clock`
  (instantiate with an integer `power`; reduce to `X`/`Z` at `dim=2,
  power=1`), and the two-qubit {py:data}`Sum` singleton (generalized
  controlled add). {py:class}`SwapLevels` transposes two basis levels;
  {py:data}`Fourier`/{py:data}`Fourierdg` are the qudit DFT (the `H`
  analogue for any dimension); {py:class}`SubspaceRX`/{py:class}`SubspaceRY`/
  {py:class}`SubspaceRZ` embed a qubit rotation into a chosen 2-level
  subspace; and the two-qudit {py:class}`CClock` (instantiate with an
  integer `power`) is a generalized controlled-phase that reduces to `CZ`
  at `dim=2, power=1`.
- **Reset**: the {py:data}`Reset` singleton — see
  [Measurement and conditions](measurement-and-conditions.md).

## Addressing targets

A gate's targets are passed as a single operand (one-qubit gates) or a
tuple of operands (multi-qubit gates):

```python
program.add(fqc.ops.H, 0)              # single target
program.add(fqc.ops.CX, (0, 1))        # two targets, control first
```

Each operand is either a bare integer or an explicit
{py:class}`~fatqcat.RegisterRef`. A bare integer is only accepted when the
program has exactly one register of the relevant kind — with multiple
quantum registers, address a specific one explicitly:

```python
program = fqc.Program([fqc.QuantumRegister(2, name="a"), fqc.QuantumRegister(2, name="b")])
program.add(fqc.ops.H, program.qreg[1][0])   # qubit 0 of register "b"
```

## Example: qudit gates on qutrits

The dimension-generic gates work the same way on any `dim`, not just `dim=2`.
This program builds two qutrits, exercises `Fourier`/`Fourierdg` as a
round-trip identity, moves a basis level with `SwapLevels`, rotates it back
with `SubspaceRX`, and applies a controlled phase with `CClock`:

```python
import numpy as np
import fatqcat as fqc

qreg = fqc.QuantumRegister(2, dim=3)   # two qutrits
creg = fqc.ClassicalRegister(2, dim=3)
program = fqc.Program([qreg], [creg])

program.add(fqc.ops.Fourier, 0)                    # qudit Hadamard analogue...
program.add(fqc.ops.Fourierdg, 0)                   # ...immediately undone
program.add(fqc.ops.SwapLevels(0, 2), 0)            # |0> -> |2>
program.add(fqc.ops.SubspaceRX(np.pi, (0, 2)), 0)   # |2> -> |0> (up to global phase)
program.add(fqc.ops.CClock(1), (0, 1))              # phase-only here: qubit 1 stays |0>
program.measure_all()

result = fqc.backends.StateVectorBackend().run(program, shots=100).result()
print(result.get_counts_as_tuples())               # {(0, 0): 100}
```
