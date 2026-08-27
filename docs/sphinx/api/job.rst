Job
===

``Job`` is eager and generic in its completed payload. Ordinary execution
uses ``Job[Result]``; parameter sweeps use the same runtime class with an
ordered ``Job[list[Result]]`` payload. This does not introduce a separate job
lifecycle or sweep-job type.

Backends return jobs already in a terminal state. Read
:py:attr:`~fatqat.Job.status` to inspect that state and call
:py:meth:`~fatqat.Job.result` to obtain the payload. A failed job re-raises its
stored execution error from ``result()``. Applications receive jobs from
``run(...)`` or ``run_sweep(...)`` rather than constructing them directly.

.. autoclass:: fatqat.Job
   :members: status, result
   :show-inheritance:
