PhaseDamping
============

.. currentmodule:: fatqat.noise

:class:`PhaseDamping` removes coherence without transferring population. Its
``p`` simulator channel and ``rate``/``t_phi`` emulator Lindblad operator use
different multilevel conventions, so choose the form required by the target
backend rather than relying on an implicit conversion.

Choose a form
-------------

Supply exactly one keyword:

.. list-table:: Parameterizations
   :header-rows: 1
   :widths: 18 48 34

   * - Argument
     - Meaning
     - Constraint
   * - ``p``
     - Full-dephasing weight for one simulator channel application
     - Finite real in ``[0, 1]``
   * - ``rate``
     - Rate scaling the local dephasing Lindblad operator
     - Finite nonnegative real in inverse time
   * - ``t_phi``
     - Pure-dephasing time, normalized to ``rate = 1 / t_phi``
     - Finite positive real in the backend's time unit

``t_phi`` is a convenient way to supply a rate. The object stores
``rate = 1 / t_phi`` rather than keeping ``t_phi`` as a separate value.

Simulators
----------

For a target of dimension :math:`d`, probability mode implements

.. math::

   \mathcal{E}_p(\rho)
   = (1-p)\rho + p\,\operatorname{diag}(\operatorname{diag}(\rho)).

All populations are unchanged, and every off-diagonal entry is multiplied by
:math:`1-p`, regardless of the separation between its two levels. The channel
acts on one selected operand.

Pulse emulators
---------------

The ``rate`` and ``t_phi`` forms use the local Lindblad operator

.. math::

   L=\sqrt{2r}\,\operatorname{diag}(0,1,\ldots,d-1).

Its effect on coherence between levels j and k is

.. math::

   \rho_{jk}(t)=e^{-r(j-k)^2t}\rho_{jk}(0).

See :ref:`noise-emulator-support` for the accepted scopes and implementation-map
requirements of each built-in emulator.

Converting between forms
------------------------

:meth:`PhaseDamping.as_probability` uses :math:`p=1-e^{-rt}`, and
:meth:`PhaseDamping.as_rate` uses :math:`r=-\log(1-p)/t`. These relations match
a qubit channel and adjacent-level coherence. For larger systems, the two
forms differ: the simulator channel damps every coherence uniformly, while
Lindblad evolution scales the decay with :math:`(j-k)^2`.

Duration must be finite and nonnegative. Probability 1 has no finite rate, and
a nonzero probability cannot be converted at zero duration.

API
---

.. autoclass:: PhaseDamping
   :members: as_probability, as_rate
   :show-inheritance:
