<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# API 参考


本参考文档用于查阅 FATQAT 的公共类、函数和异常。如需了解完整工作流程，请先阅读[构建、绘制并运行一个 Program](../guide/quickstart.md)。

!!! note "Python API 文档"

    本节的说明文字已翻译为中文；由 mkdocstrings 直接从 Python 源码生成的类、函数、参数和返回值说明仍保留英文，以确保与当前代码中的 docstring 完全一致。

!!! info "Simulator 与 Emulator"

    本中文文档统一将门级、矩阵型的 `Simulator` 后端称为“模拟器”，将解析脉冲并演化物理动力学的 `Emulator` 后端称为“仿真器”。

## 构建程序

- [Program](program.md)
- [寄存器](registers.md)
- [操作](operations.md)
- [脉冲控制](pulse-control/index.md)

## 运行与检查

- [模拟器（Simulator）](simulators/index.md)
- [仿真器（Emulator）](emulators/index.md)
- [噪声](noise.md)
- [可观测量与估计](estimator.md)
- [Job](job.md)
- [Result](result.md)

## 集成与扩展

- [互操作](interoperability/index.md)
- [矩阵实现](implementation.md)

## 错误与稳定性

- [异常](exceptions.md)
- [实验性 API](experimental.md)
