<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 异常


如果同一个处理器能够从任意 FATQAT 错误中恢复，请捕获 [`FatqatError`][fatqat.errors.FatqatError]；如果恢复方式取决于具体故障，则捕获相应子类。无效的参数类型和值也可能改为抛出 Python 的 `TypeError` 或 `ValueError`。

[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] 是 [`BackendValidationError`][fatqat.errors.BackendValidationError] 的子类。后端通常会在 `run` 返回作业之前抛出验证错误。之后发生的执行错误可能保存在返回的 [`Job`][fatqat.Job] 中，并由 [`result`][fatqat.Job.result] 抛出。

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
