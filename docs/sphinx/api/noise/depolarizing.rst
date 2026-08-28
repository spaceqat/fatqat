Depolarizing
============

.. currentmodule:: fatqat.noise

:class:`Depolarizing` describes uniform mixing with the maximally mixed state.
Choose ``p`` for a simulator channel or ``rate`` for Lindblad operators on a
compatible pulse emulator. Backends do not convert between the two forms.

Choose a form
-------------

.. list-table:: Parameterizations
   :header-rows: 1
   :widths: 18 42 16 24

   * - Form
     - Meaning
     - Applies to
     - Typical backend
   * - ``p``
     - Complete-depolarization weight for one channel application, in
       ``[0, 1]``
     - The selected operands
     - Matrix simulator
   * - ``rate``
     - Finite, nonnegative rate scaling local depolarization Lindblad operators
     - One subsystem
     - Lindblad-capable pulse emulator

Exactly one form is required. A rate is measured in the inverse of the
selected backend's time unit. For example, a backend whose durations are in
microseconds interprets the rate in inverse microseconds.

Simulators
----------

For selected operands with combined dimension :math:`d`, probability mode is

.. math::

   \mathcal{E}_p(\rho) = (1-p)\rho + p\frac{I_d}{d}.

The channel acts jointly on all selected operands. Applying
``Depolarizing(p=...)`` to two qubits therefore uses :math:`d=4`; it does not
create two independent single-qubit channels. For independent local noise,
use ``target_positions`` or separate registrations.

Here ``p`` is the weight of complete depolarization, not the probability of
choosing a nonidentity error. For one qubit, the total probability assigned to
the nonidentity branches is :math:`3p/4`.

On supported methods, ``statevector`` samples one Kraus branch per application,
``density_matrix`` applies the exact Kraus sum, and ``superop`` constructs the
complete channel. See :ref:`noise-simulator-support` for restrictions.

Pulse emulators
---------------

For local dimension :math:`d`, rate mode is normalized so its Lindblad
generator is

.. math::

   \mathcal{L}_r(\rho)
   = r\left(\operatorname{Tr}(\rho)\frac{I_d}{d}-\rho\right).

Over duration :math:`t`, the corresponding probability parameter is

.. math::

   p(t)=1-e^{-rt}.

:meth:`Depolarizing.as_probability` and :meth:`Depolarizing.as_rate` perform
this parameter conversion when you know the duration. A duration must be
finite and nonnegative. Probability 1 has no finite rate, and a nonzero
probability cannot be converted at zero duration.

Rate mode requires a registered Lindblad implementation. See
:ref:`noise-emulator-support` for built-in and custom-map availability.

API
---

.. autoclass:: Depolarizing
   :members: as_probability, as_rate, num_subsystems
   :show-inheritance:
