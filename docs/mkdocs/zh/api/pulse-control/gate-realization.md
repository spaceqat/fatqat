
# 门实现


[`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] 将普通门映射到脉冲定义。直接 [PulseOperation](pulse-operation.md) 已包含自身的控制，不使用此映射。

## 规则


规则以 `rule(operation, *, device_operands=...)` 形式调用，必须返回 [`PulseDefinition`][fatqat.emulator.PulseDefinition]。可以注册一条通用规则来处理每个有序设备操作数元组，也可以为特定元组分别注册规则。元组专用条目还可以是固定定义，或只接受 `operation` 的可调用对象。元组包含 `("q0", "q1")` 等有序物理标签，而不是程序寄存器引用。

## 定义


一个 [`PulseDefinition`][fatqat.emulator.PulseDefinition] 包含持续时间、控制元组和可选的 `PhaseShift` 或 `PhaseSwap` 动作。条件和噪声仍保留在程序中的操作上。

`PhaseShift` 在脉冲后改变一个模型参考系。`PhaseSwap` 交换两个参考系。直接脉冲操作没有后置动作。

使用门时，仿真器会调用并验证所选规则。请抛出 [`BackendValidationError`][fatqat.errors.BackendValidationError] 来报告不受支持的操作数或参数。其他异常以及非 `PulseDefinition` 返回值会报告为 `PulseImplementationError`。

[超导量子比特](../pulse-emulator.md)和[中性原子](../atom-emulators.md)页面展示内置映射及完整工作流程。

## 参考


::: fatqat.emulator.PulseImplementationMap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PulseDefinition
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PhaseShift
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.PhaseSwap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
