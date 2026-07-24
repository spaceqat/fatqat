# Gates

Use gates from the ``fq.ops`` namespace with {py:meth}`~fatqat.Program.add`.
The
{doc}`operations API reference <../api/operations>` contains exact
signatures and matrices; this page focuses on the everyday calling pattern.

## Fixed gates and parametric gates

Fixed gates are ready-to-use values. Do not add parentheses:

```python
import fatqat as fq

program = fq.Program(2)
program.add(fq.ops.H, 0)
program.add(fq.ops.X, 1)
program.add(fq.ops.CX, (0, 1))
```

Parametric gates are classes. Create one with its parameter before adding
it. Rotation and phase angles are in radians:

```python
program.add(fq.ops.RX(0.2), 0)
program.add(fq.ops.RZ(1.5), 1)
program.add(fq.ops.CPhase(0.4), (0, 1))
```

Passing `fq.ops.RX` rather than `fq.ops.RX(0.2)` is a common mistake. The
former is the gate class; the latter is the operation you add.

## Targets and target order

A one-qubit gate takes one target. A multi-qubit gate takes one tuple of
targets:

```python
program.add(fq.ops.H, 0)
program.add(fq.ops.CX, (0, 1))
program.add(fq.ops.CCX, (0, 1, 2))
```

For controlled gates, controls come first and the final target comes last.
For example, `CX(0, 1)` uses qubit 0 as the control and qubit 1 as the
target. When multiple registers make an integer ambiguous, use a register
reference such as `program.qreg[1][0]`.

## Gate families

| Family | Operations |
| --- | --- |
| fixed single-qubit | {py:data}`~fatqat.operations.I`, {py:data}`~fatqat.operations.H`, {py:data}`~fatqat.operations.S`, {py:data}`~fatqat.operations.Sdg`, {py:data}`~fatqat.operations.T`, {py:data}`~fatqat.operations.Tdg`, {py:data}`~fatqat.operations.X`, {py:data}`~fatqat.operations.Y`, {py:data}`~fatqat.operations.Z` |
| parametric | {py:class}`~fatqat.operations.RX`, {py:class}`~fatqat.operations.RY`, {py:class}`~fatqat.operations.RZ`, {py:class}`~fatqat.operations.Phase`, {py:class}`~fatqat.operations.CPhase` |
| fixed multi-qubit | {py:data}`~fatqat.operations.CX`, {py:data}`~fatqat.operations.CZ`, {py:data}`~fatqat.operations.Swap`, {py:data}`~fatqat.operations.CY`, {py:data}`~fatqat.operations.CS`, {py:data}`~fatqat.operations.iSwap`, {py:data}`~fatqat.operations.CCX`, {py:data}`~fatqat.operations.CSwap` |
| reset | {py:data}`~fatqat.operations.Reset`; see [Measurement and conditions](measurement-and-conditions.md) |
| qudit | {py:class}`~fatqat.operations.Shift`, {py:class}`~fatqat.operations.Clock`, {py:data}`~fatqat.operations.Sum`, {py:class}`~fatqat.operations.SwapLevels`, {py:data}`~fatqat.operations.Fourier`, {py:data}`~fatqat.operations.Fourierdg`, {py:class}`~fatqat.operations.SubspaceRX`, {py:class}`~fatqat.operations.SubspaceRY`, {py:class}`~fatqat.operations.SubspaceRZ`, {py:class}`~fatqat.operations.CClock` |

## Optional grid selections

A {py:class}`~fatqat.GridRegister` can name a row, column, block, or all of its qubits. The
view-capable operations are `RX`, `RY`, `RZ`, `CX`, and `CZ`.

```python
import fatqat as fq

atoms = fq.GridRegister(2, 3, name="atoms")
program = fq.Program([atoms])
program.add(fq.ops.RX(0.2), atoms.row(1))
program.add(fq.ops.CX, (atoms.row(0), atoms.row(1)))
```

The two views in the `CX` example are paired in order: the first entry in
row 0 controls the first entry in row 1, and so on. The views must have the
same length. The backend validates any device-specific constraints when the
program runs.

## Qudit gates

Qudit gates work with registers whose `dim` is greater than 2. `Shift` and
`Clock` generalize `X` and `Z`; `Sum` generalizes `CX`; and `Fourier`
generalizes `H`. See [Advanced user topics](advanced.md) for a complete
qutrit example.
