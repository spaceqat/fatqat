Experimental APIs
=================

The APIs listed here are public, but their signatures and behavior may change
between releases. Use them only when the stable backend workflow does not
cover your integration.

Direct construction
-------------------

Backends return :py:class:`~fatqat.Job` and :py:class:`~fatqat.Result`;
applications normally should not construct either class. When writing a
backend integration, create a successful job as ``Job("DONE", result=value)``
or a failed job as ``Job("ERROR", error=exc)``, and translate the backend's
output into a ``Result``. See :doc:`job` and :doc:`result` for the interfaces.

Extension points
----------------

The following extension points are also evolving:

- :doc:`implementation` for custom matrices and device-specific gate rules
- :doc:`pulse-control/gate-realization` for custom pulse definitions

Built-in pulse maps are listed under :doc:`emulators/index`.
