---
title: "PauliChannel"
---

# PauliChannel


[`PauliChannel`][fatqat.noise.PauliChannel] represents a stochastic mixture of Pauli-string
errors. It is useful for biased or correlated qubit noise that uniform
[`Depolarizing`][fatqat.noise.Depolarizing] cannot express.

## Terms


Pass either a mapping or an iterable of `(string, probability)` pairs. Every
string must:

* be nonempty;
* contain only uppercase `I`, `X`, `Y`, and `Z`;
* have the same width as every other string.

Each probability is an `int` or `float` other than `bool` and must be
finite and in `[0, 1]`. Duplicate strings are an error. Nonidentity
probabilities may sum to less than 1; FATQAT assigns the remaining weight to
the all-identity string. An explicitly supplied identity term must agree with
that implied value. Small floating-point round-off is tolerated.

FATQAT consumes the input and stores `PauliChannel.terms` as an immutable
tuple. The identity term comes first; the other terms keep their input order:

```python
import fatqat as fq

channel = fq.noise.PauliChannel({"X": 0.01, "Z": 0.02})
assert channel.terms == (("I", 0.97), ("X", 0.01), ("Z", 0.02))
```

## Simulators


After the identity weight has been filled in, the channel is

$$
\mathcal{E}(\rho) = \sum_i p_i P_i \rho P_i.
$$

Compatible simulators apply this channel after the matched operation. See
[Simulators](backend-support.md#noise-simulator-support) for built-in availability and scope.

## Target order


The string width determines the number of targets, and every target must be a
qubit. Strings follow Qiskit's displayed Pauli-string convention, the same
reading as [`Observable`][fatqat.Observable] labels: the rightmost character
describes the first target. For targets `(q0, q1)`:

**Two-qubit ordering**

| String | First target `q0` | Second target `q1` |
| --- | --- | --- |
| `XI` | `I` | `X` |
| `IX` | `X` | `I` |

FATQAT checks the target count and qubit dimensions when the program
runs.

## API


::: fatqat.noise.PauliChannel
    options:
      members:
        - "num_subsystems"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
