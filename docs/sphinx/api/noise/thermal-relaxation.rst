ThermalRelaxation
=================

.. currentmodule:: fatqat.noise

:class:`ThermalRelaxation` models zero-temperature T1 and T2 relaxation on an
emulator. It combines downward population decay with the additional pure
dephasing needed to reproduce T2. For a qubit simulator, convert it explicitly
for a known duration with :meth:`ThermalRelaxation.as_channels`.

Times and rates
---------------

``t1`` and ``t2`` are finite positive values in the same unit; they have no
intrinsic unit of their own. A duration passed to
:meth:`ThermalRelaxation.as_channels` must use that same unit. When registering
the noise with an emulator, use the emulator model's time unit. Physical
consistency requires

.. math::

   T_2 \leq 2T_1.

The derived rates are

.. math::

   \gamma_1 = \frac{1}{T_1}, \qquad
   \gamma_\phi = \frac{1}{T_2}-\frac{1}{2T_1}.

:attr:`ThermalRelaxation.amplitude_rate` returns :math:`\gamma_1`, and
:attr:`ThermalRelaxation.pure_dephasing_rate` returns :math:`\gamma_\phi`.
The latter is nonnegative under the T2 bound and becomes zero at
:math:`T_2=2T_1`.

Pulse emulators
---------------

For a :math:`d`-level pulse model, the emulator uses the local Lindblad
operators

.. math::

   \begin{aligned}
   L_1 &= \sum_{k=1}^{d-1}\sqrt{\frac{k}{T_1}}
          |k-1\rangle\!\langle k|,\\
   L_\phi &= \sqrt{2\gamma_\phi}\,
             \operatorname{diag}(0,1,\ldots,d-1).
   \end{aligned}

The second operator is omitted when :math:`\gamma_\phi=0`. This is a local,
zero-temperature model: it includes downward relaxation but no thermal
excitation or equilibrium-population parameter.

See :ref:`noise-emulator-support` for the accepted scopes and implementation-map
requirements of each built-in emulator.

Simulator conversion
--------------------

:meth:`ThermalRelaxation.as_channels` returns an ordered pair for a duration
:math:`t`:

.. math::

   \begin{aligned}
   p_1(t) &= 1-e^{-t/T_1},\\
   p_\phi(t) &= 1-e^{-\gamma_\phi t}.
   \end{aligned}

Apply the returned :class:`AmplitudeDamping` first and :class:`PhaseDamping`
second. For a qubit, their composition gives population decay
:math:`e^{-t/T_1}` and
coherence decay :math:`e^{-t/T_2}`.

This conversion is for qubits. It returns one amplitude probability, whereas
a higher-dimensional amplitude-damping channel needs :math:`d-1` values.
``ThermalRelaxation`` itself is not a simulator channel. See
:ref:`noise-simulator-support` for simulator support.

API
---

.. autoclass:: ThermalRelaxation
   :members: amplitude_rate, pure_dephasing_rate, as_channels
   :show-inheritance:
