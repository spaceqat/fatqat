---
title: "Job"
---

# Job

!!! warning "Job API under development"

    The Job API is under active development. Its interfaces and supported
    behavior may change between releases. Pin an exact FatQat version when
    reproducibility matters.

FATQAT simulators, emulators, and [`Estimator`][fatqat.Estimator] return a completed
[`Job`][fatqat.Job]. Call
[`result`][fatqat.Job.result] to obtain the result; it does not wait.

The [LQCloud adapter](interoperability/lqcloud.md) returns the cloud SDK's
asynchronous job instead; that job follows the SDK's status and result API.

::: fatqat.Job
    options:
      members:
        - "status"
        - "result"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
