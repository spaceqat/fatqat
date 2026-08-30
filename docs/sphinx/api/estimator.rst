Observables and estimation
===========================

Use :class:`~fatqat.Estimator` to evaluate one or more
:class:`~fatqat.Observable` values from a backend's final state. The program
must be unmeasured, fully bound, and qubit-only; the backend must return a
statevector or density matrix in the program's logical qubit space.

Estimate an observable
----------------------

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   program = fq.Program(2)
   program.add(ops.H, 0)
   program.add(ops.CX, (0, 1))

   estimator = fq.Estimator(fq.simulator.Simulator("SV"))
   observable = fq.Observable([("ZZ", 1.0)])
   result = estimator.run(program, observable).result()
   expectation = result.get_expectation()

For a guided workflow, see :doc:`../guide/interpret-results`.

Construct an observable
-----------------------

Dense labels put qubit 0 at the right and accept ``I``, ``X``, ``Y``, and
``Z``. These forms are equivalent:

.. code-block:: python

   fq.Observable([("ZZ", 1.5)])
   fq.Observable(["ZZ"], coeffs=[1.5])

:meth:`~fatqat.Observable.from_sparse` names each non-identity factor and its
qubit explicitly. It also supports ``ZERO`` and ``ONE`` projectors:

.. code-block:: python

   fq.Observable.from_sparse(
       [(["ONE", "Z"], (5, 3), 1.5)],
       num_qubits=6,
   )

Coefficients must be real.

Exact and sampled results
-------------------------

Pass one observable to receive scalar expectation and standard-error values,
or a list or tuple to receive arrays in the same order.

``shots=0`` computes an exact value. A positive ``shots`` value samples each
observable term, and :meth:`~fatqat.Result.get_std` reports the resulting
standard error. Set ``simulation_config["seed"]`` to reproduce a sampled run.

Configure the simulation method, runtime, and noise on the backend. Invalid
programs, observable widths, and shot values raise
:class:`~fatqat.errors.BackendValidationError` before a job is returned;
unsupported observable types raise ``TypeError``. Later failures are raised by
:meth:`~fatqat.Job.result`.

Use a density-matrix backend when the program resets qubits or channel noise
applies.

For a program with ``N`` qubits, the returned statevector must have shape
``(2**N,)`` or the density matrix must have shape ``(2**N, 2**N)``. A backend
state with another shape raises :class:`~fatqat.errors.BackendValidationError`
before any Pauli expectation kernel runs. In particular, a full physical qutrit
state returned by a Transmon emulator is not implicitly projected into the
logical subspace. Inspect that physical state directly with the backend
:class:`~fatqat.Result` until explicit leakage-aware observable semantics are
available.

Read estimator results with :meth:`~fatqat.Result.get_expectation` and
:meth:`~fatqat.Result.get_std`. Run the backend separately if you also need its
final state.

Parameter sweeps
----------------

:meth:`~fatqat.Estimator.run_sweep` evaluates binding rows in input order.
Validation errors raise directly; other row failures are raised by
:meth:`~fatqat.Job.result`. No partial result list is returned. See
:doc:`../guide/simulation` for a guided parameter sweep; accepted binding
shapes and seed behavior are specified here.

Detailed reference
------------------

.. autoclass:: fatqat.Observable
   :members:
   :show-inheritance:

.. autoclass:: fatqat.Estimator
   :members:
   :show-inheritance:
