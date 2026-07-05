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

## Matrix convention

For every matrix below, the basis is ordered $|0\rangle, |1\rangle, \dots$
per qubit, and for multi-qubit gates the operand passed *first* to
{py:meth}`~qnsim.Program.add` is the most-significant tensor factor — e.g.
for a two-qubit gate with targets $(a, b)$, row/column index $2a + b$
(basis order $|00\rangle, |01\rangle, |10\rangle, |11\rangle$). For
controlled gates the control operand(s) always come first, so this is the
same as saying the control is the more-significant qubit in the matrix.

## Gate categories

### Single-qubit fixed gates

$$
I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \qquad
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \qquad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} \qquad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

$$
H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \qquad
S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} \qquad
S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}
$$

$$
T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} \qquad
T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}
$$

{py:data}`I`, {py:data}`H`, {py:data}`S`, {py:data}`Sdg`, {py:data}`T`,
{py:data}`Tdg`, {py:data}`X`, {py:data}`Y`, {py:data}`Z`.

### Parametric gates

Instantiate with an angle $\theta$ in radians: {py:class}`RX`,
{py:class}`RY`, {py:class}`RZ`, {py:class}`Phase`, and the two-qubit
{py:class}`CPhase`.

$$
R_X(\theta) = \begin{pmatrix}
\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
-i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
\end{pmatrix} \qquad
R_Y(\theta) = \begin{pmatrix}
\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
\end{pmatrix}
$$

$$
R_Z(\theta) = \begin{pmatrix}
e^{-i\theta/2} & 0 \\
0 & e^{i\theta/2}
\end{pmatrix} \qquad
\mathrm{Phase}(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}
$$

Note that {py:class}`RZ` and {py:class}`Phase` differ by the global phase
$e^{-i\theta/2}$ — they act identically up to that phase.

$$
\mathrm{CPhase}(\theta) = \begin{pmatrix}
1&0&0&0\\ 0&1&0&0\\ 0&0&1&0\\ 0&0&0&e^{i\theta}
\end{pmatrix}
$$

`targets = (control, target)`; the phase $e^{i\theta}$ is applied only to
$|11\rangle$.

### Multi-qubit fixed gates

{py:data}`CX`, {py:data}`CZ`, {py:data}`Swap`, {py:data}`CY`, {py:data}`CS`,
{py:data}`iSwap`, {py:data}`CCX`, {py:data}`CSwap`. For controlled gates,
operand order is `(control, ..., target)`.

$$
CX = \begin{pmatrix}
1&0&0&0\\ 0&1&0&0\\ 0&0&0&1\\ 0&0&1&0
\end{pmatrix} \qquad
CZ = \begin{pmatrix}
1&0&0&0\\ 0&1&0&0\\ 0&0&1&0\\ 0&0&0&-1
\end{pmatrix} \qquad
\mathrm{Swap} = \begin{pmatrix}
1&0&0&0\\ 0&0&1&0\\ 0&1&0&0\\ 0&0&0&1
\end{pmatrix}
$$

$$
CY = \begin{pmatrix}
1&0&0&0\\ 0&1&0&0\\ 0&0&0&-i\\ 0&0&i&0
\end{pmatrix} \qquad
CS = \begin{pmatrix}
1&0&0&0\\ 0&1&0&0\\ 0&0&1&0\\ 0&0&0&i
\end{pmatrix} \qquad
i\mathrm{Swap} = \begin{pmatrix}
1&0&0&0\\ 0&0&i&0\\ 0&i&0&0\\ 0&0&0&1
\end{pmatrix}
$$

{py:data}`CCX` (Toffoli, `targets = (control0, control1, target)`) and
{py:data}`CSwap` (Fredkin, `targets = (control, target0, target1)`) act on
three qubits, basis order $|000\rangle, \dots, |111\rangle$:

$$
CCX = \begin{pmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0\\
0&0&0&0&1&0&0&0\\
0&0&0&0&0&1&0&0\\
0&0&0&0&0&0&0&1\\
0&0&0&0&0&0&1&0
\end{pmatrix} \qquad
CSwap = \begin{pmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0\\
0&0&0&0&1&0&0&0\\
0&0&0&0&0&0&1&0\\
0&0&0&0&0&1&0&0\\
0&0&0&0&0&0&0&1
\end{pmatrix}
$$

Equivalently: $CCX$ flips the target iff both controls are $|1\rangle$, and
$CSwap$ exchanges the two targets iff the control is $|1\rangle$.

### Dimension-generic (qudit) gates

{py:class}`Shift` and {py:class}`Clock` (instantiate with an integer
`power`; reduce to `X`/`Z` at `dim=2, power=1`), and the two-qubit
{py:data}`Sum` singleton (generalized controlled add). These gates build
their matrix from the target subsystem's dimension $d$ at backend lowering,
so there is no single fixed-size matrix — the general formulas are:

$$
\mathrm{Shift}(d, p) : |k\rangle \mapsto |(k + p) \bmod d\rangle \qquad
\mathrm{Clock}(d, p) : |k\rangle \mapsto \omega^{kp} |k\rangle,\ \ \omega = e^{2\pi i/d}
$$

$$
\mathrm{Sum} : |i, j\rangle \mapsto |i,\, (i + j) \bmod d\rangle
$$

For example, at $d = 3$:

$$
\mathrm{Shift}(3, 1) = \begin{pmatrix}
0&0&1\\ 1&0&0\\ 0&1&0
\end{pmatrix} \qquad
\mathrm{Clock}(3, 1) = \begin{pmatrix}
1&0&0\\ 0&\omega&0\\ 0&0&\omega^2
\end{pmatrix},\ \ \omega = e^{2\pi i/3}
$$

And {py:data}`Sum` at its smallest dimension, $d = 2$ (`targets =
(control, target)`), which reduces to exactly {py:data}`CX`:

$$
\mathrm{Sum}\big|_{d=2} = \begin{pmatrix}
1&0&0&0\\ 0&1&0&0\\ 0&0&0&1\\ 0&0&1&0
\end{pmatrix}
$$

### Reset

The {py:data}`Reset` singleton — see
[Measurement and conditions](measurement-and-conditions.md). Reset is
non-unitary (a repreparation in $|0\rangle$), so it has no fixed matrix.

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
</content>
