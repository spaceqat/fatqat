Qiskit
======

.. py:module:: fatqat.qiskit

Use :class:`FatqatBackend` to run supported Qiskit circuits on FATQAT, or
:func:`circuit_to_program` to convert a circuit directly. Install Qiskit
separately; Qiskit Aer is not required.

To use FATQAT in a Qiskit backend workflow:

.. code-block:: python

   from qiskit import QuantumCircuit, generate_preset_pass_manager
   from fatqat.qiskit import FatqatBackend

   circuit = QuantumCircuit(2, 2)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.measure([0, 1], [0, 1])

   backend = FatqatBackend()
   pass_manager = generate_preset_pass_manager(backend=backend)
   isa_circuit = pass_manager.run(circuit)
   job = backend.run(isa_circuit, shots=100, seed_simulator=7)
   counts = job.result().get_counts()

Circuit conversion
------------------

:func:`circuit_to_program` accepts bound circuits containing instructions from
:func:`build_simulator_target`. Transpile other circuits to that target first.
Named registers and circuit metadata are preserved. The global phase is stored
in ``program.metadata`` but is not applied to the simulated state.

Measurement and reset are converted, while barriers are discarded. Unsupported
instructions, dynamic control flow, unbound instruction parameters, and an
unbound global phase raise :exc:`QiskitConversionError`.

Backend execution
-----------------

:class:`FatqatBackend` defaults to ``method="statevector"`` and
``runtime="numpy"``. ``method`` accepts the names and aliases supported by
:class:`~fatqat.simulator.Simulator`; ``runtime`` accepts ``"numpy"`` or
``"numba"``, case-insensitively. Unsupported values produce an error job from
``run()`` rather than failing construction. The backend accepts a FATQAT
:class:`~fatqat.NoiseModel`, not a Qiskit Aer noise model.

``run()`` accepts one circuit or a nonempty iterable and supports three
options:

- ``shots``: positive ``int``, default ``1024``;
- ``memory``: ``bool``, default ``False``; and
- ``seed_simulator``: non-negative ``int`` or ``None``, default ``None``.

Options passed to ``run()`` override ``backend.options``. Invalid options, an
empty circuit iterable, or an iterable containing non-circuit values raise
:exc:`QiskitBackendError` before a job is created. If circuit conversion or
execution fails later, ``run()`` returns an ``ERROR`` job and
:meth:`FatqatJob.result` raises Qiskit's ``QiskitError``.

The result contains Qiskit-formatted counts, including for multiple classical
registers. ``memory=True`` returns entries consistent with those counts,
without guaranteeing shot order. Circuits without classical bits have no
counts or memory data. Statevector and related simulator artifacts are not
included in the Qiskit result.

Jobs and provider helper
------------------------

Execution is synchronous, so :class:`FatqatJob` is ``DONE`` or ``ERROR`` when
returned. :class:`FatqatProvider` creates configured :class:`FatqatBackend`
instances for code that expects a provider-style interface.

Reference
---------

.. py:function:: circuit_to_program(circuit)

   Convert a bound, static Qiskit ``QuantumCircuit`` into a
   :class:`~fatqat.Program`. Unrepresentable circuits raise
   :exc:`QiskitConversionError`; non-circuit inputs raise :exc:`TypeError`.

.. py:class:: FatqatBackend(*, method="statevector", runtime="numpy", noise_model=None, provider=None, name="fatqat_simulator")

   Synchronous Qiskit ``BackendV2`` backed by the gate-level FATQAT simulator.

   .. py:method:: run(run_input, **run_options)

      Execute one circuit or a nonempty iterable and return a completed
      :class:`FatqatJob`.

   .. py:attribute:: target

      The supported instruction basis used for transpilation.

   .. py:attribute:: max_circuits

      Always ``None``; batches are not capped.

   .. py:attribute:: coupling_map

      Always ``None``; circuits are not restricted to a coupling map.

.. py:class:: FatqatJob

   Completed Qiskit ``JobV1`` returned by :meth:`FatqatBackend.run`.

   .. py:method:: status()

      Return Qiskit's ``DONE`` or ``ERROR`` status.

   .. py:method:: result(timeout=None)

      Return the Qiskit ``Result`` or raise ``QiskitError``. ``timeout`` is
      accepted and ignored.

   .. py:method:: submit()

      Return immediately because the job has already run.

   .. py:method:: cancel()

      Return ``False``.

   .. py:method:: backend()

      Return the creating backend.

.. py:class:: FatqatProvider(**default_backend_kwargs)

   Provider-style helper that creates configured :class:`FatqatBackend`
   instances.

   .. py:method:: backends(name=None, *, filters=None, **kwargs)

      Return a new backend in a one-item list when ``name`` matches, otherwise
      an empty list. ``filters`` has no effect; other constructor options in
      ``kwargs`` override the provider defaults.

   .. py:method:: get_backend(name=None, **kwargs)

      Return a new matching backend, or raise :exc:`ValueError`.

.. py:function:: build_simulator_target()

   Return a new unbounded Qiskit target for the supported gate basis.

.. py:exception:: QiskitConversionError

   A circuit cannot be converted to a FATQAT program. This is a
   :exc:`fatqat.errors.FatqatError`.

.. py:exception:: QiskitBackendError

   A run request was rejected before execution. This is a
   :exc:`fatqat.errors.FatqatError`.
