Backends (``fq.backends``)
==========================

A backend validates and executes a :py:class:`~fatqat.Program`. Normal applications use a
backend through :py:meth:`run <~fatqat.backends.SimulatorBackend.run>` and read the resulting :py:class:`~fatqat.Result`; they do not
need to customize its implementation map. For custom matrix rules or
device-specific maps, see :doc:`experimental`.

General-purpose simulator
-------------------------

:py:class:`~fatqat.backends.SimulatorBackend` (``method="statevector", noise=None``)

- ``method="statevector"`` is the default pure-state simulator. ``"SV"``
  is an accepted alias.
- ``method="density_matrix"`` simulates exact mixed states. ``"DM"`` is
  an accepted alias.
- Pass a :py:class:`~fatqat.NoiseModel` with ``noise=...`` to run the same program with
  channel or readout noise.

Run a program
-------------

:py:meth:`run <~fatqat.backends.SimulatorBackend.run>` (``program, *, shots=1024, result_config=None, seed=None``)

- ``shots`` controls how many samples are collected when counts are
  requested.
- ``result_config`` selects ``counts`` and the method’s native state field.
- ``seed`` makes sampled results reproducible.
- ``run(...)`` returns a :py:class:`~fatqat.Job`. Call ``job.result()`` to obtain the
  :py:class:`~fatqat.Result`. Its status and failure lifecycle controls are described in
  :doc:`experimental` as an evolving API.

See :doc:`../guide/running-and-results` for complete counts, statevector,
and density-matrix examples. The optional parallel-shot settings are
documented in :doc:`../guide/advanced`.

Constrained simulated targets
-----------------------------

:py:class:`~fatqat.backends.SCQubitIBMSimulator`,
:py:class:`~fatqat.backends.SCQubitGoogleSimulator`, and
:py:class:`~fatqat.backends.FakeAtomGridBackend` are optional
simulated targets with fixed native-gate and connectivity constraints. Use
them when those constraints are part of an experiment or test, not as the
default backend for a first program.

Detailed reference
------------------

.. autoclass:: fatqat.backends.SimulatorBackend
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.SCQubitIBMSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.SCQubitGoogleSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.FakeAtomGridBackend
   :members:
   :show-inheritance:
