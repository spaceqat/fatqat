<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 实验性 API


此处列出的 API 已公开，但其签名和行为可能在不同版本间发生变化。只有当稳定的后端工作流程无法满足集成需求时，才应使用这些 API。

## 直接构造


后端会返回 [`Job`][fatqat.Job] 和 [`Result`][fatqat.Result]；应用程序通常不应自行构造这两个类。编写后端集成时，可用 `Job("DONE", result=value)` 创建成功的作业，用 `Job("ERROR", error=exc)` 创建失败的作业，并将后端输出转换成 `Result`。接口说明参阅 [Job](job.md) 和 [Result](result.md)。

## 扩展点


以下扩展点也仍在演进：

- [矩阵实现](implementation.md)：自定义矩阵和设备专用门规则
- [门实现](pulse-control/gate-realization.md)：自定义脉冲定义

内置脉冲映射列于[仿真器](emulators/index.md)一节。
