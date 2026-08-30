Result
======

A :class:`~fatqat.Result` contains only the fields produced by one run.
Inspect :attr:`~fatqat.Result.available_data` when a field is optional, then
use its accessor. An unavailable accessor raises
:class:`~fatqat.errors.ResultFieldUnavailableError` instead of returning
``None``.

.. list-table:: Result fields
   :header-rows: 1
   :widths: 24 31 45

   * - Field
     - Accessor
     - Produced by
   * - ``"counts"``
     - :meth:`~fatqat.Result.get_counts` or
       :meth:`~fatqat.Result.get_counts_as_tuples`
     - A backend run with measurement counts
   * - ``"statevector"``
     - :meth:`~fatqat.Result.get_statevector`
     - A statevector run with final-state output enabled
   * - ``"density_matrix"``
     - :meth:`~fatqat.Result.get_density_matrix`
     - A density-matrix run with final-state output enabled
   * - ``"unitary"``
     - :meth:`~fatqat.Result.get_unitary`
     - A unitary run with final-state output enabled
   * - ``"superop"``
     - :meth:`~fatqat.Result.get_superop`
     - A super-operator run with final-state output enabled
   * - ``"expectation"`` and ``"std"``
     - :meth:`~fatqat.Result.get_expectation` and
       :meth:`~fatqat.Result.get_std`
     - An :class:`~fatqat.Estimator` run
   * - Backend extension name
     - :meth:`~fatqat.Result.get_data`
     - A backend extension

``"final_state"`` is a request name, not an available-data name. A produced
state or operator uses its concrete representation name from the table.
Deterministic unmeasured runs enable method-native output by default. A
stochastic pulse-emulator state—caused by measurement, statevector reset, or
potentially active statevector Lindblad evolution—must be requested explicitly
with ``shots=1``.

Ordering and mutable values
---------------------------

:meth:`~fatqat.Result.get_counts` returns a new dictionary of display strings.
The highest-index classical slot is on the left and slot 0 is on the right. If
any classical dimension is at least 10, commas make multi-digit outcomes
unambiguous. :meth:`~fatqat.Result.get_counts_as_tuples` instead puts flat
classical slot 0 at tuple position 0.

Most other accessors return the value stored in the result. Copy arrays or
dictionaries before changing them if you need to preserve the original.
Metadata records the normalized ``simulation_config`` and ``result_config``.
Backend extensions may add fields. Pulse-emulator metadata records the
canonical ``method``; any solver diagnostics are informational and their keys
are not a public compatibility contract. Keep the model, arrangement,
controls, and application metadata alongside a result when they are needed to
reproduce a physical run.

For every complete state or operator, ``metadata["state_axes"]`` lists the
physical subsystems from least to most significant. Each entry contains a
``device_operand`` and its program ``register_ref``; ``register_ref`` is
``None`` when a physical model contains a subsystem the Program did not
address. Position 0 is the least-significant subsystem of a flat basis index.
For local dimensions ``dims``, position ``q`` has place value
``prod(dims[:q])``. Density-matrix rows and columns use the same basis order.

A counts-only run zero-fills every declared classical slot that was never
written by measurement and emits a standard ``UserWarning``. This usually
indicates a missing measurement.

See :doc:`../guide/interpret-results` for the guided interpretation workflow;
the conventions above are the canonical state-axis and count-order contract.

Detailed reference
------------------

.. autoclass:: fatqat.Result
   :members:
   :show-inheritance:
