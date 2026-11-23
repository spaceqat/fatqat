---
title: "Exceptions"
---

# Exceptions

Catch [`FatqatError`][fatqat.errors.FatqatError] when you handle FatQat's
execution and conversion errors in the same way. Catch a specific subclass when
the action depends on what failed. An invalid argument type or value can also
raise Python's `TypeError` or `ValueError`.

## Run and result errors { #errors-when-running-a-program }

- **Before execution.**
  [`BackendValidationError`][fatqat.errors.BackendValidationError] means the
  backend cannot run the request as supplied.
  [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] is the
  subclass for an operation or feature the backend does not support.

- **During execution.** The backend's error settings determine whether the run
  call raises an execution error directly or returns a failed [`Job`][fatqat.Job].
  For a failed job, [`result()`][fatqat.Job.result] raises the stored error.

- **Reading results.**
  [`ResultFieldUnavailableError`][fatqat.errors.ResultFieldUnavailableError] means
  the requested field is absent from the result. Use `available_data` to see what
  the run produced.

For conversion errors, see [OpenQASM](interoperability/openqasm.md#errors) and
[Qiskit](interoperability/qiskit.md#backend-execution). The
[compiler](compiler.md) has a separate exception hierarchy.

## API reference

::: fatqat.errors.FatqatError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.errors.BackendValidationError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.errors.BackendExecutionError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.errors.UnsupportedOperationError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.errors.MatrixImplementationError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.errors.PulseImplementationError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.errors.ResultFieldUnavailableError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
