---
title: "TransitionRelaxation"
---

# TransitionRelaxation

[`TransitionRelaxation`][fatqat.noise.TransitionRelaxation] describes exactly
one explicitly authored local jump operator. It is independent of hardware
kind: the same interface can describe a transmon ladder, one atomic decay
branch, or another finite-dimensional relaxation mechanism.

Each coefficient key is `(source, destination)`, and one descriptor defines

$$
A = \sum_{(s,d)} c_{sd}|d\rangle\!\langle s|.
$$

FATQAT validates nonnegative level indices at construction and checks them
against the physical target dimension during backend preparation. Numerical
index order does not by itself determine energy order; the caller identifies
the physically relaxing source and destination states.

## Shared and separate jumps

Terms in one descriptor are components of one coherent jump operator. For
example, a truncated bosonic lowering operator for a transmon-like model can
contain both ladder transitions, while two uncorrelated atom-like decay
branches use separate descriptors:

```python
from math import sqrt
import fatqat as fq

transmon_relaxation = fq.noise.TransitionRelaxation(
    rate=0.001,
    coefficients={(1, 0): 1.0, (2, 1): sqrt(2.0)},
)

atom_decay_to_zero = fq.noise.TransitionRelaxation(
    rate=0.03,
    coefficients={(2, 0): 1.0},
)
atom_decay_to_one = fq.noise.TransitionRelaxation(
    rate=0.02,
    coefficients={(2, 1): 1.0},
)
```

In rate mode, the two atom descriptors become separate collapse operators
and dissipators, with no cross terms. In finite mode, they become sequential
channel applications in registration order. Putting both atom transitions
in one descriptor would instead make them components of one coherent jump
and can produce coherence between the destination states.

This distinction is explicit. FATQAT does not infer it from labels such as
`atom` or `transmon`, and it does not insert harmonic-oscillator
coefficients. Supply `1`, `sqrt(2)`, or measured matrix elements directly.

Coefficient scale is not normalized: rescaling every coefficient by
$\alpha\ne0$ and the strength by $1/|\alpha|^2$ leaves the channel or
dissipator unchanged, subject to the finite-form bounds below.

## Finite channel form

Use a scalar `p` in `[0, 1]` for one elementary Kraus-channel application
on a matrix simulator.

Each probability-form declaration resolves independently:

$$
K_1 = \sqrt{p}A, \qquad
K_0 = \sqrt{I-K_1^\dagger K_1}.
$$

The operator under the square root must be positive semidefinite. When
several declarations match the same target, the simulator applies their
channels sequentially in authored (registration) order. For two
unit-coefficient branches from the same source level, survival is
$(1-p_1)(1-p_2)$ rather than $1-p_1-p_2$, a discrepancy of $p_1p_2$.
Destination populations can also depend on registration order, with those
differences beginning at second order. Sequential composition therefore
agrees with simultaneous independent decay only to first order for small
probabilities. Use rate mode on a pulse emulator when accurate continuous
competing-decay dynamics matter.

`p` is a dimensionless jump strength rather than necessarily the total jump
probability. For a basis state $|s\rangle$, the jump probability is

$$
\Pr(\mathrm{jump}\mid |s\rangle)
= p\lVert A|s\rangle\rVert^2
= p\sum_d |c_{sd}|^2.
$$

## Continuous form

Use a finite, nonnegative scalar `rate` on a compatible pulse emulator. One
descriptor resolves to

$$
L = \sqrt{\mathrm{rate}}A.
$$

Multiple descriptors remain separate collapse operators, so the generator is
$\sum_\mu \mathcal D[L_\mu]$, not
$\mathcal D[\sum_\mu L_\mu]$. Rates use the inverse of the emulator model's
time unit.

The `p` and `rate` forms share the same authored operator convention but
are not general finite-time conversions of one another. The finite form is an
elementary jump/no-jump channel; continuous evolution may include repeated or
cascaded jumps.

See [Backend support](backend-support.md) for accepted forms and scopes.

## API

::: fatqat.noise.TransitionRelaxation
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
