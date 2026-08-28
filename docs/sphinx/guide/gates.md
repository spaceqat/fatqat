# Gates

Import ``fatqat.operations`` as ``ops`` and pass operations to
{py:meth}`~fatqat.Program.add`. The
{doc}`operations API reference <../api/operations>` gives exact signatures and
matrices; this page shows the usual calling patterns.

## Fixed and parameterized gates

Fixed gates are ready-to-use values. Do not add parentheses:

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2)
program.add(ops.H, 0)
program.add(ops.X, 1)
program.add(ops.CX, (0, 1))
```

Parameterized gates are classes. Create one with its parameter before adding
it. Rotation and phase angles are in radians:

```python
program.add(ops.RX(0.2), 0)
program.add(ops.RZ(1.5), 1)
program.add(ops.CPhase(0.4), (0, 1))
```

## Targets and target order

A one-qubit gate takes one target. A multi-qubit gate takes one tuple of
targets:

```python
three_qubit_program = fq.Program(3)
three_qubit_program.add(ops.H, 0)
three_qubit_program.add(ops.CX, (0, 1))
three_qubit_program.add(ops.CCX, (0, 1, 2))
```

For controlled gates, controls come first and the final target comes last.
For example, `program.add(ops.CX, (0, 1))` uses qubit 0 as the control and
qubit 1 as the target. When multiple registers make an integer ambiguous,
index the intended register, such as `right[0]`.

## Gate families

| Family | Operations |
| --- | --- |
| fixed single-qubit | {py:data}`~fatqat.operations.I`, {py:data}`~fatqat.operations.H`, {py:data}`~fatqat.operations.S`, {py:data}`~fatqat.operations.Sdg`, {py:data}`~fatqat.operations.SX`, {py:data}`~fatqat.operations.T`, {py:data}`~fatqat.operations.Tdg`, {py:data}`~fatqat.operations.X`, {py:data}`~fatqat.operations.Y`, {py:data}`~fatqat.operations.Z` |
| parametric | {py:class}`~fatqat.operations.RX`, {py:class}`~fatqat.operations.RY`, {py:class}`~fatqat.operations.RZ`, {py:class}`~fatqat.operations.Phase`, {py:class}`~fatqat.operations.U`, {py:class}`~fatqat.operations.U1`, {py:class}`~fatqat.operations.U2`, {py:class}`~fatqat.operations.U3`, {py:class}`~fatqat.operations.CPhase` |
| fixed multi-qubit | {py:data}`~fatqat.operations.CX`, {py:data}`~fatqat.operations.CZ`, {py:data}`~fatqat.operations.Swap`, {py:data}`~fatqat.operations.CY`, {py:data}`~fatqat.operations.CS`, {py:data}`~fatqat.operations.iSwap`, {py:data}`~fatqat.operations.CCX`, {py:data}`~fatqat.operations.CSwap` |
| reset | {py:data}`~fatqat.operations.Reset`; see [Measurement and conditions](measurement-and-conditions.md) |
| qudit | {py:class}`~fatqat.operations.Shift`, {py:class}`~fatqat.operations.Clock`, {py:data}`~fatqat.operations.Sum`, {py:class}`~fatqat.operations.SwapLevels`, {py:data}`~fatqat.operations.Fourier`, {py:data}`~fatqat.operations.InverseFourier`, {py:class}`~fatqat.operations.SubspaceRX`, {py:class}`~fatqat.operations.SubspaceRY`, {py:class}`~fatqat.operations.SubspaceRZ`, {py:class}`~fatqat.operations.CClock` |

## Optional grid selections

A {py:class}`~fatqat.GridRegister` can name a row, column, block, or all of its
qubits. The view-capable operations are `RX`, `RY`, `RZ`, `CX`, and `CZ`.

```python
import fatqat as fq
import fatqat.operations as ops

qubits = fq.GridRegister(2, 3, name="qubits")
program = fq.Program([qubits])
program.add(ops.RX(0.2), qubits.row(1))
program.add(ops.CX, (qubits.row(0), qubits.row(1)))
```

The two views in the `CX` example are paired in order: the first entry in row
0 controls the first entry in row 1, and so on. Both views must use the same
kind of grid selection and have the same length. Views on one grid must not
overlap.
The backend validates any device-specific constraints when the program runs.

## Qudit gates

Dimension-generic gates are mainly useful for registers with `dim > 2`, but
they are defined for every `dim >= 2`. `Shift` and `Clock` generalize `X` and
`Z`; `Sum` generalizes `CX`; and `Fourier` generalizes `H`. See
[Advanced](advanced.md) for a complete qutrit example.
