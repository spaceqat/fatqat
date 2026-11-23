---
title: "Qubit gates"
---

# Qubit gates


These gates act on qubits (dimension 2). Choose the gate and operand order
from the tables below. For grouped targets, see
[`Program.add()`][fatqat.Program.add]. The backend checks dimensions and
device support when the program runs.

## Fixed gates


Pass fixed gates such as `ops.H` directly to `add()`.

**Fixed single-qubit gates**

| Value | Basis action |
| --- | --- |
| [`I`][fatqat.operations.I] | Leaves $\lvert 0\rangle$ and $\lvert 1\rangle$ unchanged. |
| [`H`][fatqat.operations.H] | Maps $\lvert 0\rangle$ to $(\lvert 0\rangle+\lvert 1\rangle)/\sqrt{2}$ and $\lvert 1\rangle$ to $(\lvert 0\rangle-\lvert 1\rangle)/\sqrt{2}$. |
| [`X`][fatqat.operations.X] | Exchanges $\lvert 0\rangle$ and $\lvert 1\rangle$. |
| [`Y`][fatqat.operations.Y] | Maps $\lvert 0\rangle$ to $i\lvert 1\rangle$ and $\lvert 1\rangle$ to $-i\lvert 0\rangle$. |
| [`Z`][fatqat.operations.Z] | Maps $\lvert 1\rangle$ to $-\lvert 1\rangle$. |
| [`S`][fatqat.operations.S] | Maps $\lvert 1\rangle$ to $i\lvert 1\rangle$. |
| [`Sdg`][fatqat.operations.Sdg] | Maps $\lvert 1\rangle$ to $-i\lvert 1\rangle$. |
| [`SX`][fatqat.operations.SX] | Two applications have the same action as X. |
| [`T`][fatqat.operations.T] | Maps $\lvert 1\rangle$ to $e^{i\pi/4}\lvert 1\rangle$. |
| [`Tdg`][fatqat.operations.Tdg] | Maps $\lvert 1\rangle$ to $e^{-i\pi/4}\lvert 1\rangle$. |

**Fixed multi-qubit gates**

| Value | Target order | Basis action |
| --- | --- | --- |
| [`CX`][fatqat.operations.CX] | `(control, target)` | Applies X to the target when the control is $\lvert 1\rangle$. |
| [`CY`][fatqat.operations.CY] | `(control, target)` | Applies Y to the target when the control is $\lvert 1\rangle$. |
| [`CZ`][fatqat.operations.CZ] | `(control, target)` | Negates $\lvert 11\rangle$. |
| [`CS`][fatqat.operations.CS] | `(control, target)` | Applies S to the target when the control is $\lvert 1\rangle$. |
| [`Swap`][fatqat.operations.Swap] | `(target0, target1)` | Exchanges the two target states. |
| [`iSwap`][fatqat.operations.iSwap] | `(target0, target1)` | Maps $\lvert 01\rangle$ to $i\lvert 10\rangle$ and $\lvert 10\rangle$ to $i\lvert 01\rangle$. |
| [`CCX`][fatqat.operations.CCX] | `(control0, control1, target)` | Toffoli: applies X when both controls are $\lvert 1\rangle$. |
| [`CSwap`][fatqat.operations.CSwap] | `(control, target0, target1)` | Fredkin: exchanges the two targets when the control is $\lvert 1\rangle$. |

## Parameterized gates


All angles are in radians and are not normalized. Every angle field accepts a
[`Parameter`][fatqat.Parameter] for later binding through
[`fatqat.Program.assign_parameters`][fatqat.Program.assign_parameters].

**Parameterized qubit gates**

| Constructor | Targets | Definition |
| --- | --- | --- |
| [`RX`][fatqat.operations.RX] `(theta)` | One scalar or one view | Rotation about the X axis by `theta`. |
| [`RY`][fatqat.operations.RY] `(theta)` | One scalar or one view | Rotation about the Y axis by `theta`. |
| [`RZ`][fatqat.operations.RZ] `(theta)` | One scalar or one view | Rotation about the Z axis by `theta`. |
| [`Phase`][fatqat.operations.Phase] `(theta)` | One scalar or one view | Differs from RZ only by global phase. |
| [`U`][fatqat.operations.U] `(theta, phi, lam)` | One scalar or one view | General single-qubit gate using Qiskit's parameter convention. |
| [`U1`][fatqat.operations.U1] `(lam)` | One scalar or one view | Equivalent to `Phase(lam)`. |
| [`U2`][fatqat.operations.U2] `(phi, lam)` | One scalar or one view | Equivalent to `U(pi/2, phi, lam)`. |
| [`U3`][fatqat.operations.U3] `(theta, phi, lam)` | One scalar or one view | Same matrix as `U(theta, phi, lam)`; retained for Qiskit compatibility. |
| [`CPhase`][fatqat.operations.CPhase] `(theta)` | `(control, target)` scalars or compatible views | Multiplies $\lvert 11\rangle$ by $e^{i\theta}$. |

## API reference


Common operation properties are documented on the [Operations overview](../operations.md).

### Fixed values


::: fatqat.operations.I
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.H
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.X
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.Y
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.Z
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.S
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.Sdg
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.SX
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.T
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.Tdg
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.CX
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.CY
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.CZ
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.CS
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.Swap
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.iSwap
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.CCX
    options:
      heading_level: 4
      show_attribute_values: false

::: fatqat.operations.CSwap
    options:
      heading_level: 4
      show_attribute_values: false

### Parameterized classes


::: fatqat.operations.RX
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.RY
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.RZ
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.Phase
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U1
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U2
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U3
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.CPhase
    options:
      heading_level: 4
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

## Matrix definitions

### Fixed gate matrices

These matrices act on column state vectors. For the single-qubit gates, rows
and columns use basis order
$(|0\rangle,|1\rangle)$:

$$
\begin{aligned}
I &= \begin{pmatrix}1&0\\0&1\end{pmatrix},
& H &= \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\\[0.5em]
X &= \begin{pmatrix}0&1\\1&0\end{pmatrix},
& Y &= \begin{pmatrix}0&-i\\i&0\end{pmatrix},\\[0.5em]
Z &= \begin{pmatrix}1&0\\0&-1\end{pmatrix},
& S &= \begin{pmatrix}1&0\\0&i\end{pmatrix},\\[0.5em]
\mathrm{Sdg} &= \begin{pmatrix}1&0\\0&-i\end{pmatrix},
& \mathrm{SX} &= \frac{1}{2}\begin{pmatrix}1+i&1-i\\1-i&1+i\end{pmatrix},\\[0.5em]
T &= \begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix},
& \mathrm{Tdg} &= \begin{pmatrix}1&0\\0&e^{-i\pi/4}\end{pmatrix}.
\end{aligned}
$$

For each two-qubit matrix, the targets are $(q_0,q_1)$ and rows and
columns use basis order
$(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$. The first operand
$q_0$ is the local most-significant bit. It is the control for `CX`,
`CY`, `CZ`, and `CS`; $q_1$ is the target.

$$
CX = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&1\\
0&0&1&0
\end{pmatrix}
$$

$$
CY = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&-i\\
0&0&i&0
\end{pmatrix}
$$

$$
CZ = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&-1
\end{pmatrix}
$$

$$
CS = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&i
\end{pmatrix}
$$

For `Swap` and `iSwap`, the same basis order applies with operand order
`(target0, target1)`.

$$
\mathrm{Swap} = \begin{pmatrix}
1&0&0&0\\
0&0&1&0\\
0&1&0&0\\
0&0&0&1
\end{pmatrix}
$$

$$
i\mathrm{Swap} = \begin{pmatrix}
1&0&0&0\\
0&0&i&0\\
0&i&0&0\\
0&0&0&1
\end{pmatrix}
$$

For `CCX`, operand order is `(control0, control1, target)`. Rows and
columns use basis order $(|000\rangle,|001\rangle,|010\rangle, |011\rangle,|100\rangle,|101\rangle,|110\rangle,|111\rangle)$, with the
first operand as the most-significant bit:

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
\end{pmatrix}
$$

For `CSwap`, operand order is `(control, target0, target1)`. Rows and
columns use the same three-bit basis order, again with the first operand as
the most-significant bit:

$$
\mathrm{CSwap} = \begin{pmatrix}
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

<a id="matrix-definitions_1"></a>

### Parameterized gate matrices

These matrices act on column state vectors. The single-qubit matrices below
use row and column basis order
$(|0\rangle,|1\rangle)$. Let
$c=\cos(\theta/2)$ and $s=\sin(\theta/2)$:

$$
RX(\theta) = \begin{pmatrix}
c&-is\\
-is&c
\end{pmatrix},
\qquad
RY(\theta) = \begin{pmatrix}
c&-s\\
s&c
\end{pmatrix}
$$

$$
RZ(\theta) = \begin{pmatrix}
e^{-i\theta/2}&0\\
0&e^{i\theta/2}
\end{pmatrix},
\qquad
\mathrm{Phase}(\theta) = \begin{pmatrix}
1&0\\
0&e^{i\theta}
\end{pmatrix}
$$

For `U`, `U1`, `U2`, and `U3`, operands still use the single-qubit
basis above and the parameter order is the constructor order shown in the
table:

$$
U(\theta,\phi,\lambda) = \begin{pmatrix}
\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
\end{pmatrix}
$$

$$
U1(\lambda) = \begin{pmatrix}
1&0\\
0&e^{i\lambda}
\end{pmatrix},
\qquad
U2(\phi,\lambda) = \frac{1}{\sqrt{2}}\begin{pmatrix}
1&-e^{i\lambda}\\
e^{i\phi}&e^{i(\phi+\lambda)}
\end{pmatrix}
$$

$$
U3(\theta,\phi,\lambda) = \begin{pmatrix}
\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
\end{pmatrix}
$$

For `CPhase`, operand order is `(control, target)`. Rows and columns use
basis order $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$, with the
control as the local most-significant bit:

$$
\mathrm{CPhase}(\theta) = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&e^{i\theta}
\end{pmatrix}
$$
