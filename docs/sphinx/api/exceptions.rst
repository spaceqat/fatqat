Exceptions
==========

Catch :py:class:`~fatqat.errors.FatqatError` when one handler can recover from
any FATQAT error, or catch a subclass when recovery depends on the failure.
Invalid argument types and values may instead raise Python's ``TypeError`` or
``ValueError``.

:py:class:`~fatqat.errors.UnsupportedOperationError` is a subtype of
:py:class:`~fatqat.errors.BackendValidationError`. A backend normally raises
validation errors before ``run`` returns a job. A later execution failure may
be stored on the returned :py:class:`~fatqat.Job` and raised by
:py:meth:`~fatqat.Job.result`.

.. autoexception:: fatqat.errors.FatqatError
   :no-members:
   :no-inherited-members:

.. autoexception:: fatqat.errors.BackendValidationError
   :no-members:
   :no-inherited-members:

.. autoexception:: fatqat.errors.BackendExecutionError
   :no-members:
   :no-inherited-members:

.. autoexception:: fatqat.errors.UnsupportedOperationError
   :no-members:
   :no-inherited-members:

.. autoexception:: fatqat.errors.MatrixImplementationError
   :no-members:
   :no-inherited-members:

.. autoexception:: fatqat.errors.PulseImplementationError
   :no-members:
   :no-inherited-members:

.. autoexception:: fatqat.errors.ResultFieldUnavailableError
   :no-members:
   :no-inherited-members:
