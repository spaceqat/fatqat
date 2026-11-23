---
title: "Job"
---

# Job

!!! warning "Job API under development"

    The Job API is under active development. Its interfaces and supported
    behavior may change between releases. Pin an exact FatQat version when
    reproducibility matters.

A [`Job`][fatqat.Job] holds the outcome of a submission. Jobs returned by built-in
backends and [`Estimator`][fatqat.Estimator] are already complete. Call
[`result()`][fatqat.Job.result] to read the value or raise the execution error
stored in the job.

A successful single run returns one [`Result`][fatqat.Result] through `result()`.
A successful sweep returns a list in input order. Invalid inputs can raise before
a job is returned. Whether an execution error is raised immediately or stored in a
job depends on the backend's error-handling option.

## API reference

::: fatqat.Job
    options:
      members:
        - "status"
        - "result"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
