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
state uses its concrete representation name from the table. Deterministic runs
enable final-state output by default.

Ordering and mutable values
---------------------------

:meth:`~fatqat.Result.get_counts` returns a new dictionary of display strings.
The highest-index classical slot is on the left and slot 0 is on the right. If
any classical dimension is at least 10, commas make multi-digit outcomes
unambiguous. :meth:`~fatqat.Result.get_counts_as_tuples` instead puts flat
classical slot 0 at tuple position 0.

Most other accessors return the value stored in the result. Copy arrays or
dictionaries before changing them if you need to preserve the original.
Metadata contents depend on the backend or estimator that produced the result.
See :doc:`../guide/running-and-results` for state and operator basis ordering.

Detailed reference
------------------

.. autoclass:: fatqat.Result
   :members:
   :show-inheritance:
