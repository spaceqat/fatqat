Job
===

``Job`` is eager and generic in its completed payload. Ordinary execution
uses ``Job[Result]``; parameter sweeps use the same runtime class with an
ordered ``Job[list[Result]]`` payload. This does not introduce a separate job
lifecycle or sweep-job type.

.. autoclass:: fatqat.Job
   :members:
   :show-inheritance:
