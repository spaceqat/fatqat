---
title: "Exceptions"
---

# Exceptions


Catch [`FatqatError`][fatqat.errors.FatqatError] when one handler can recover from
any FATQAT error, or catch a subclass when recovery depends on the failure.
Invalid argument types and values may instead raise Python's `TypeError` or
`ValueError`.

[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] is a subtype of
[`BackendValidationError`][fatqat.errors.BackendValidationError]. A backend normally raises
validation errors before `run` returns a job. A later execution failure may
be stored on the returned [`Job`][fatqat.Job] and raised by
[`result`][fatqat.Job.result].

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
