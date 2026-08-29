---
title: "Qudit gates"
---

# Qudit gates


These gates are defined for local dimensions `d >= 2`; the table gives any
additional dimension rules. [`add`][fatqat.Program.add] records the
operation, and the backend checks support when the program runs.
Integer `power` values are reduced modulo the relevant target dimension, so
negative and oversized values are valid.

**Qudit gates**

| Value or constructor | Targets and constraints | Basis action |
| --- | --- | --- |
| [`Shift`][fatqat.operations.Shift] `(power)` | One scalar; any `d >= 2` | `\|k> -> \|(k + power) mod d>`. `Shift(1)` is X for `d=2`. |
| [`Clock`][fatqat.operations.Clock] `(power)` | One scalar; any `d >= 2` | `\|k> -> omega**(k*power)\|k>`, `omega=exp(2*pi*i/d)`. `Clock(1)` is Z for `d=2`. |
| [`Sum`][fatqat.operations.Sum] | `(control, target)` with equal dimensions | `\|i,j> -> \|i,(i+j) mod d>`. It is CX for `d=2`. |
| [`SwapLevels`][fatqat.operations.SwapLevels] `(j, k)` | One scalar; `0 <= j,k < d` and `j != k` | Exchanges `\|j>` and `\|k>` and fixes every other level. |
| [`Fourier`][fatqat.operations.Fourier] | One scalar; any `d >= 2` | `\|j> -> sum(exp(2*pi*i*j*k/d)\|k>) / sqrt(d)`. It is H for `d=2`. |
| [`InverseFourier`][fatqat.operations.InverseFourier] | One scalar; any `d >= 2` | Conjugate transpose of `Fourier`; uses the negative exponent. |
| [`SubspaceRX`][fatqat.operations.SubspaceRX] `(theta, (j, k))` | One scalar; two distinct in-range levels | With `c=cos(theta/2)`, `s=sin(theta/2)`: `\|j> -> c\|j>-i*s\|k>` and `\|k> -> -i*s\|j>+c\|k>`. |
| [`SubspaceRY`][fatqat.operations.SubspaceRY] `(theta, (j, k))` | One scalar; two distinct in-range levels | `\|j> -> c\|j>+s\|k>` and `\|k> -> -s\|j>+c\|k>`. Reversing `(j, k)` reverses the rotation. |
| [`SubspaceRZ`][fatqat.operations.SubspaceRZ] `(theta, (j, k))` | One scalar; two distinct in-range levels | `\|j>` gains `exp(-i*theta/2)` and `\|k>` gains `exp(i*theta/2)`. Reversing the pair reverses the rotation. |
| [`CClock`][fatqat.operations.CClock] `(power)` | `(control, target)`; dimensions may differ | `\|i,j>` gains `omega**(i*j*power)` using the target's `omega=exp(2*pi*i/d_target)`. It is CZ for two qubits and power 1. |

Level pairs must contain distinct, non-negative integers. Equality and
negativity are checked at construction, and the target dimension is checked
when the operation is added. `Sum` requires equal control and target
dimensions; the backend rejects a mismatch when the program runs.

## Matrix definitions


The matrices below act on column vectors. For one-qudit gates, rows and
columns use the computational-basis order
$\lvert 0\rangle,\lvert 1\rangle,\ldots,\lvert d-1\rangle$.

### Shift and Clock


For `Shift(power=p)` and `Clock(power=p)`, let
$\omega_d=\exp(2\pi i/d)$. Their general operators are

$$
X_d^p
= \sum_{k=0}^{d-1}
  \left\lvert (k+p)\bmod d \right\rangle\!\left\langle k\right\rvert,
\qquad
Z_d^p
= \sum_{k=0}^{d-1}
  \omega_d^{pk}\left\lvert k\right\rangle\!\left\langle k\right\rvert.
$$

### Sum


`Sum` takes operands as `(control, target)`. For equal dimension `d`,
the control is the local most-significant factor:

$$
\operatorname{SUM}_d
= \sum_{i,j=0}^{d-1}
  \left\lvert i,(i+j)\bmod d\right\rangle
  \!\left\langle i,j\right\rvert.
$$

### SwapLevels


For distinct levels $j$ and $k$, the general operator is

$$
S_{j,k}
= I - \lvert j\rangle\!\langle j\rvert
    - \lvert k\rangle\!\langle k\rvert
    + \lvert j\rangle\!\langle k\rvert
    + \lvert k\rangle\!\langle j\rvert.
$$

### Fourier transforms


With $\omega_d=\exp(2\pi i/d)$, `Fourier` and `InverseFourier`
are

$$
F_d
= \frac{1}{\sqrt d}\sum_{j,k=0}^{d-1}
  \omega_d^{jk}\lvert k\rangle\!\langle j\rvert,
\qquad
F_d^{-1}=F_d^\dagger
= \frac{1}{\sqrt d}\sum_{j,k=0}^{d-1}
  \omega_d^{-jk}\lvert k\rangle\!\langle j\rvert.
$$

### Subspace rotations


Let $c=\cos(\theta/2)$ and $s=\sin(\theta/2)$. For the ordered
level pair `subspace=(j, k)`, the general operators are

$$
\begin{aligned}
R_X^{(j,k)}(\theta)
&= I + (c-1)(\lvert j\rangle\!\langle j\rvert
                +\lvert k\rangle\!\langle k\rvert)
   -is(\lvert j\rangle\!\langle k\rvert
                +\lvert k\rangle\!\langle j\rvert), \\
R_Y^{(j,k)}(\theta)
&= I + (c-1)(\lvert j\rangle\!\langle j\rvert
                +\lvert k\rangle\!\langle k\rvert)
   -s\lvert j\rangle\!\langle k\rvert
   +s\lvert k\rangle\!\langle j\rvert, \\
R_Z^{(j,k)}(\theta)
&= I + (e^{-i\theta/2}-1)\lvert j\rangle\!\langle j\rvert
   +(e^{i\theta/2}-1)\lvert k\rangle\!\langle k\rvert.
\end{aligned}
$$

### CClock


`CClock(power=p)` takes operands as `(control, target)`. If their
dimensions are $d_c$ and $d_t$, the control is the local
most-significant factor, $\omega_t=\exp(2\pi i/d_t)$, and

$$
\operatorname{CClock}_{d_c,d_t}^{(p)}
= \sum_{i=0}^{d_c-1}\sum_{j=0}^{d_t-1}
  \omega_t^{ijp}\lvert i,j\rangle\!\langle i,j\rvert.
$$

## API reference


Common operation properties are documented on the [Operations overview](../operations.md).

::: fatqat.operations.Shift
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.Clock
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.Sum
    options:
      show_attribute_values: false

::: fatqat.operations.SwapLevels
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.Fourier
    options:
      show_attribute_values: false

::: fatqat.operations.InverseFourier
    options:
      show_attribute_values: false

::: fatqat.operations.SubspaceRX
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.SubspaceRY
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.SubspaceRZ
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.CClock
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
