<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# Job


FATQAT 原生后端和 [`Estimator`][fatqat.Estimator] 返回已经完成的 [`Job`][fatqat.Job]。调用 [`result`][fatqat.Job.result] 即可取得结果；该调用不会等待。

::: fatqat.Job
    options:
      members:
        - "status"
        - "result"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
