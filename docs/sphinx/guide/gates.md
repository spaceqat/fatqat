# Gates

All gates live in the `fq.ops` namespace. The full list with signatures is
in the [API reference](../api/operations.rst); this page covers how to use
them.

```{currentmodule} fatqat.operations
```

## Singletons vs. classes

Fixed (parameter-free) gates are exported as ready-to-use singleton values:

```python
program.add(fq.ops.H, 0)
program.add(fq.ops.X, 0)
```

Parametric gates are exported as classes and must be instantiated with their
parameter before use:

```python
program.add(fq.ops.RX(0.2), 0)
program.add(fq.ops.CPhase(1.5), (0, 1))
```

Passing an unparameterized class where a value is expected (e.g.
`fq.ops.RX` instead of `fq.ops.RX(0.2)`) raises a {py:exc}`TypeError` from
{py:meth}`~fatqat.Program.add` that names the mistake.

### Importing gates directly

`fq.ops.H` is fine for the odd gate here and there, but for a long circuit
with hundreds of `program.add(...)` calls, repeating the `fq.ops.` prefix
gets tedious. Every gate name is a plain top-level import, so it's fine to
pull out just the ones a script uses:

```python
from fatqat.operations import H, CX, RZ

program.add(H, 0)
program.add(CX, (0, 1))
program.add(RZ(0.3), 1)
```

Be careful with this for the single-letter gates (`I`, `H`, `S`, `T`, `X`,
`Y`, `Z`) — they're easy to shadow with an unrelated local variable of the
same name (a matrix called `I`, a set called `S`, and so on), which fails
silently rather than raising. Import only what a given script needs, and
avoid `from fatqat.operations import *` for exactly this reason.

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
program.add(fq.ops.H, 0)              # single target
program.add(fq.ops.CX, (0, 1))        # two targets, control first
```

Each operand is either a bare integer or an explicit
{py:class}`~fatqat.RegisterRef`. A bare integer is only accepted when the
program has exactly one register of the relevant kind — with multiple
quantum registers, address a specific one explicitly:

```python
program = fq.Program([fq.QuantumRegister(2, name="a"), fq.QuantumRegister(2, name="b")])
program.add(fq.ops.H, program.qreg[1][0])   # qubit 0 of register "b"
```

## Qudit gates

The dimension-generic gates work the same way on any `dim`, not just `dim=2`.
This program builds two qutrits, exercises {py:data}`Fourier`/{py:data}`Fourierdg`
as a round-trip identity, moves a basis level with {py:class}`SwapLevels`,
rotates it back with {py:class}`SubspaceRX`, and applies a controlled phase
with {py:class}`CClock`:

```python
import numpy as np
import fatqat as fq

qreg = fq.QuantumRegister(2, dim=3)   # two qutrits
creg = fq.ClassicalRegister(2, dim=3)
program = fq.Program([qreg], [creg])

program.add(fq.ops.Fourier, 0)                    # qudit Hadamard analogue...
program.add(fq.ops.Fourierdg, 0)                   # ...immediately undone
program.add(fq.ops.SwapLevels(0, 2), 0)            # |0> -> |2>
program.add(fq.ops.SubspaceRX(np.pi, (0, 2)), 0)   # |2> -> |0> (up to global phase)
program.add(fq.ops.CClock(1), (0, 1))              # phase-only here: qubit 1 stays |0>
program.measure_all()

result = fq.backends.StateVectorBackend().run(program, shots=100).result()
print(result.get_counts_as_tuples())               # {(0, 0): 100}
```
