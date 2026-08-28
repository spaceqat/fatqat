<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# PulseOperation


[`PulseOperation`][fatqat.operations.PulseOperation] 会向程序添加一个显式脉冲块。它从 `fatqat.operations` 导入，通常写作 `ops.PulseOperation`。

普通门通过向 `Program.add` 传递逻辑目标来添加。对于 `PulseOperation`，不要传入目标：每个 [`PulseControl`][fatqat.emulator.PulseControl] 都已指明要驱动的物理通道。应使用 `program.add(operation)` 添加。[`ResourceLayout`][fatqat.ResourceLayout] 不会重新映射其中的通道。

## 条件与噪声


`TransmonEmulator` 和 `Atom3LevelEmulator` 允许使用 `program.add(operation, condition=...)`。若条件为假，控制会被跳过，但脉冲块仍占用其完整持续时间。在此期间，模型漂移和背景 Lindblad 噪声仍会继续作用。`Atom2LevelEmulator` 不支持条件。

不能把操作域噪声附加到直接脉冲块，因此 `noise.add(..., operation=ops.PulseOperation)` 会引发 `ValueError`。按目标或设备标签选择的背景噪声仍然适用。

## 支持情况


[脉冲控制](index.md)列出的三个脉冲仿真器支持 `PulseOperation`。[矩阵模拟器及其设备配置](../simulators/index.md)不支持，电路绘制和 OpenQASM 导出同样不支持。

## 参考


::: fatqat.operations.PulseOperation
    options:
      members: false
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
