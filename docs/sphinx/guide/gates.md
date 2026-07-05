# Gates

All gates live in the `qs.ops` namespace. The full list with signatures is
in the [API reference](../api/operations.rst); this page covers how to use
them.

```{currentmodule} qnsim.operations
```

## Singletons vs. classes

Fixed (parameter-free) gates are exported as ready-to-use singleton values:

```python
program.add(qs.ops.H, 0)
program.add(qs.ops.X, 0)
```

Parametric gates are exported as classes and must be instantiated with their
parameter before use:

```python
program.add(qs.ops.RX(0.2), 0)
program.add(qs.ops.CPhase(1.5), (0, 1))
```

Passing an unparameterized class where a value is expected (e.g.
`qs.ops.RX` instead of `qs.ops.RX(0.2)`) raises a {py:exc}`TypeError` from
{py:meth}`~qnsim.Program.add` that names the mistake.

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
  controlled add).
- **Reset**: the {py:data}`Reset` singleton — see
  [Measurement and conditions](measurement-and-conditions.md).

## Addressing targets

A gate's targets are passed as a single operand (one-qubit gates) or a
tuple of operands (multi-qubit gates):

```python
program.add(qs.ops.H, 0)              # single target
program.add(qs.ops.CX, (0, 1))        # two targets, control first
```

Each operand is either a bare integer or an explicit
{py:class}`~qnsim.RegisterRef`. A bare integer is only accepted when the
program has exactly one register of the relevant kind — with multiple
quantum registers, address a specific one explicitly:

```python
program = qs.Program([qs.QuantumRegister(2, name="a"), qs.QuantumRegister(2, name="b")])
program.add(qs.ops.H, program.qreg[1][0])   # qubit 0 of register "b"
```
