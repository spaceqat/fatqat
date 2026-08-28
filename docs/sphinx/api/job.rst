Job
===

Native FATQAT backends and :class:`~fatqat.Estimator` return a completed
:class:`~fatqat.Job`. Call
:meth:`~fatqat.Job.result` to obtain the result; it does not wait.

.. autoclass:: fatqat.Job
   :members: status, result
   :show-inheritance:
