Result
======

A backend creates a :py:class:`~fatqat.Result` when you call ``job.result()``. That is the
normal way to obtain results: request the output you need through
``result_config`` and then use an accessor. Direct construction is available
as an evolving API; see :doc:`experimental` before depending on it.

Read result data
----------------

- :py:meth:`~fatqat.Result.get_counts` returns a dictionary keyed by little-endian
  classical strings.
- :py:meth:`~fatqat.Result.get_counts_as_tuples` returns count keys with clbit 0 first.
- :py:meth:`~fatqat.Result.get_statevector` returns a requested statevector.
- :py:meth:`~fatqat.Result.get_density_matrix` returns a requested density matrix.
- :py:meth:`~fatqat.Result.get_unitary` returns a requested unitary matrix.
- :py:meth:`~fatqat.Result.get_superop` returns a requested super-operator matrix.
- :py:meth:`~fatqat.Result.get_expectation` and :py:meth:`~fatqat.Result.get_std`
  return an expectation value and its standard error. These come from
  :py:class:`~fatqat.Estimator` rather than from a backend run; see
  :doc:`estimator`.

``Result.available_data`` lists the fields actually produced, and
``Result.metadata`` records run context, including the effective result
configuration.

An accessor raises :py:class:`~fatqat.errors.ResultFieldUnavailableError` when its field was not
requested or cannot be produced for that run. See
:doc:`../guide/running-and-results` for recipes and bit order, or
:doc:`../guide/troubleshooting` for corrective actions.

Detailed reference
------------------

.. autoclass:: fatqat.Result
   :members:
   :show-inheritance:
