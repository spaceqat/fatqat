Observable and Estimator
========================

An :py:class:`~fatqat.Observable` is a Hermitian sum of weighted terms, each a
product of single-qubit letters. An :py:class:`~fatqat.Estimator` wraps a
constructed backend and reports that observable's expectation value on a
program's final state.

The two are separate from the counts path: build the backend as usual, then
wrap it. The backend owns the method, runtime, and noise model; the estimator
adds only the observable step. See :doc:`../guide/estimator` for a worked
introduction.

Build an observable
-------------------

Three constructors produce the same internal form:

- ``fq.Observable([(label, coefficient), ...])`` — dense little-endian labels
  over ``I``/``X``/``Y``/``Z``, e.g. ``fq.Observable([("ZZ", 1.5)])``.
- ``fq.Observable(labels, coeffs=[...])`` — the same, with labels and
  coefficients given separately.
- :py:meth:`Observable.from_sparse <fatqat.Observable.from_sparse>`
  (``data, *, num_qubits``) — name only the non-identity factors, e.g.
  ``[("XY", (3, 7), 1.5)]``. This is the practical constructor for wide
  registers and the only way to reach the ``ZERO``/``ONE`` projectors, whose
  names do not fit a single-character dense label.

Coefficients must be real: every letter is Hermitian, so the observable is
Hermitian exactly when its coefficients are. The ``2**n x 2**n`` matrix is
never built.

Run an estimator
----------------

:py:class:`~fatqat.Estimator` (``backend``) takes a constructed backend, e.g.
``fq.Estimator(fq.simulator.Simulator(method="DM", noise=noise))``.

:py:meth:`Estimator.run <fatqat.Estimator.run>`
(``program, observables, *, shots=0, simulation_config=None``) returns a
completed :py:class:`~fatqat.Job`. ``observables`` is a single observable or a
sequence of them; all are evaluated against one evolution, and the result shape
mirrors the input shape.

``shots=0`` computes the value exactly from the final state. Note this differs
from :py:meth:`Simulator.run <fatqat.simulator.Simulator.run>`, whose ``shots``
defaults to 1024. A positive ``shots`` samples, reproducing the statistical
error of a finite-shot experiment.

:py:meth:`Estimator.run_sweep <fatqat.Estimator.run_sweep>` adds a required
object-keyed binding batch after ``observables``. It returns one ordered
``list[Result]`` while preserving the ordinary scalar or array result shape
inside each element. See :doc:`../guide/parameters-and-sweeps`.

Read the result
---------------

- :py:meth:`~fatqat.Result.get_expectation` returns the value — a float for a
  single observable, an array for a sequence.
- :py:meth:`~fatqat.Result.get_std` returns the matching standard error, which
  is ``0`` for an exact run.

Both raise :py:class:`~fatqat.errors.ResultFieldUnavailableError` on a result
that did not come from an estimator run.

:py:class:`~fatqat.errors.BackendValidationError` is raised when the
expectation value would be ill-defined: a program that measures, a statevector
run carrying channel noise or ``Reset``, an observable whose width disagrees
with the program, or a non-qubit register. See
:doc:`../guide/estimator` for why each case is rejected and what to use
instead.

Detailed reference
------------------

.. autoclass:: fatqat.Observable
   :members:
   :show-inheritance:

.. autoclass:: fatqat.Estimator
   :members:
   :show-inheritance:
