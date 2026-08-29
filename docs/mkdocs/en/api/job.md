---
title: "Job"
---

# Job


Native FATQAT backends and [`Estimator`][fatqat.Estimator] return a completed
[`Job`][fatqat.Job]. Call
[`result`][fatqat.Job.result] to obtain the result; it does not wait.

::: fatqat.Job
    options:
      members:
        - "status"
        - "result"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
