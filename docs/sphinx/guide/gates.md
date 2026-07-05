# Gates

All gates live in the `qs.ops` namespace. The full list with signatures is
in the [API reference](../api/operations.rst); this page covers how to use
them.

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
`qs.ops.RX` instead of `qs.ops.RX(0.2)`) raises a `TypeError` from `add()`
that names the mistake.

## Gate categories

- **Single-qubit fixed**: `I`, `H`, `S`, `Sdg`, `T`, `Tdg`, `X`, `Y`, `Z`.
- **Parametric** (instantiate with an angle in radians): `RX`, `RY`, `RZ`,
  `Phase`, and the two-qubit `CPhase`.
- **Multi-qubit fixed**: `CX`, `CZ`, `Swap`, `CY`, `CS`, `iSwap`, `CCX`,
  `CSwap`. For controlled gates, operand order is `(control, ..., target)`.
- **Dimension-generic (qudit)**: `Shift` and `Clock` (instantiate with an
  integer `power`; reduce to `X`/`Z` at `dim=2, power=1`), and the
  two-qubit `Sum` singleton (generalized controlled add).
- **Reset**: the `Reset` singleton — see
  [Measurement and conditions](measurement-and-conditions.md).

## Addressing targets

A gate's targets are passed as a single operand (one-qubit gates) or a
tuple of operands (multi-qubit gates):

```python
program.add(qs.ops.H, 0)              # single target
program.add(qs.ops.CX, (0, 1))        # two targets, control first
```

Each operand is either a bare integer or an explicit `RegisterRef`. A bare
integer is only accepted when the program has exactly one register of the
relevant kind — with multiple quantum registers, address a specific one
explicitly:

```python
program = qs.Program([qs.QuantumRegister(2, name="a"), qs.QuantumRegister(2, name="b")])
program.add(qs.ops.H, program.qreg[1][0])   # qubit 0 of register "b"
```
