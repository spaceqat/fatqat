AmplitudeDamping
================

.. currentmodule:: fatqat.noise

:class:`AmplitudeDamping` describes adjacent-level decay toward the ground
state. Use ``p`` for a simulator channel or ``rate`` for a local Lindblad
operator on a compatible emulator, with one value for each transition
:math:`|k\rangle\rightarrow|k-1\rangle`.

Choose probabilities or rates
-----------------------------

Supply exactly one of ``p`` and ``rate``. Either accepts one real number or a
nonempty iterable; FATQAT stores the values as a tuple. For a :math:`d`-level
target, pass exactly :math:`d-1` values:

.. list-table:: Entry ordering
   :header-rows: 1
   :widths: 20 38 42

   * - Entry
     - Transition
     - Constraint
   * - ``value[0]``
     - :math:`|1\rangle\rightarrow|0\rangle`
     - Probability in ``[0, 1]`` or nonnegative rate
   * - ``value[1]``
     - :math:`|2\rangle\rightarrow|1\rangle`
     - Required only when :math:`d\geq3`
   * - ``value[d - 2]``
     - :math:`|d-1\rangle\rightarrow|d-2\rangle`
     - Last value for dimension :math:`d`

A scalar therefore works only for a two-level target. FATQAT checks the number
of values when the program runs and the target dimension is known.

Simulators
----------

For probabilities :math:`p_1,\ldots,p_{d-1}`, the simulator uses

.. math::

   \begin{aligned}
   K_0 &= |0\rangle\!\langle0|
          + \sum_{k=1}^{d-1}\sqrt{1-p_k}|k\rangle\!\langle k|,\\
   K_1 &= \sum_{k=1}^{d-1}\sqrt{p_k}|k-1\rangle\!\langle k|.
   \end{aligned}

One application moves population down by at most one adjacent level. The
channel acts on one selected operand; use ``target_positions`` to choose an
operand of a multi-operand gate. See :ref:`noise-simulator-support` for
built-in availability.

Pulse emulators
---------------

For rates :math:`r_k`, a compatible pulse backend uses the local Lindblad
operator

.. math::

   L = \sum_{k=1}^{d-1}\sqrt{r_k}|k-1\rangle\!\langle k|.

Rates are finite, nonnegative, and measured in the inverse of the backend's
time unit. See :ref:`noise-emulator-support` for each built-in emulator's local
dimension, accepted scope, and implementation-map requirements.

Converting between forms
------------------------

The conversion methods apply

.. math::

   p_k(t)=1-e^{-r_k t}

independently to each transition. This is exact for a two-level system. For
:math:`d>2`, Lindblad evolution can make several adjacent jumps during one
interval, while one simulator-channel application moves population down by at
most one level. Treat the returned tuple as a parameter conversion, not as the
exact multilevel evolution over that interval.

Duration must be finite and nonnegative. A probability of 1 has no finite
rate, and a nonzero probability cannot be converted to a finite rate at zero
duration.

API
---

.. autoclass:: AmplitudeDamping
   :members: as_probability, as_rate
   :show-inheritance:
