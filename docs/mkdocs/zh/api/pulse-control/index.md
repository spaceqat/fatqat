---
title: 脉冲控制
---
<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 脉冲控制


若要了解门校准、直接控制和调度工作流程，请先阅读[跟随 Program 进入物理动力学](../../guide/hamiltonian-emulation.md)。本节定义精确的脉冲对象及验证规则。

一个直接脉冲块由三种值构成：

* [SampledWaveform](sampled-waveform.md) 保存一个信号的采样值。
* [PulseControl](pulse-control.md) 将该信号分配给模型通道。
* [PulseOperation](pulse-operation.md) 将一个或多个控制组合成一条带时序的程序指令。

`PulseOperation` 从 `fatqat.operations` 导入，通常以 `ops.PulseOperation` 使用。`PulseControl` 和 `SampledWaveform` 位于 `fatqat.emulator`。与普通门不同，添加脉冲操作时无需指定目标，因为其中的通道已经指明要驱动的物理资源。

若要用脉冲实现普通门，请注册一个返回 [`PulseDefinition`][fatqat.emulator.PulseDefinition] 的 [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] 规则。参阅[门实现](gate-realization.md)。

## 验证


FatQat 会在构造这些值时检查采样结构和时序。无效偏移、非正的块持续时间、重复通道以及延伸到块范围之外的控制都会被拒绝。创建通道时，模型的通道工厂会检查寻址参数。

使用控制时，仿真器会检查模型兼容性、资源名称、实数或复数采样要求、振幅和持续时间限制，以及不能同时运行的通道组合。

## 内置支持


所有持续时间、偏移和采样时间都使用模型的时间单位。采样值使用通道单位，并可能被限制为实数。

**直接脉冲支持**

| 后端 | 通道 | 时间／采样单位 | `run()` 中的条件 |
| --- | --- | --- | --- |
| [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] | `drive(id)` 接受复数值；`detuning(id)` 和 `exchange(first, second)` 要求实数值。 | `ns`、`rad/ns` | 支持 |
| [`Atom3LevelEmulator`][fatqat.emulator.Atom3LevelEmulator] | `raman(site)` 和 `rydberg(site)` 接受复数值。 | `us`、`rad/us` | 支持 |
| [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] | 全局 `drive()` 接受复数值；全局 `detuning()` 要求实数值。 | `us`、`rad/us` | 不支持 |
| [`Simulator`][fatqat.simulator.Simulator] | 不支持直接脉冲操作。 | -- | -- |

[超导量子比特](../pulse-emulator.md)和[中性原子](../atom-emulators.md)页面说明模型资源和限制。

<a id="pulse-probability-noise"></a>


## 连续时间噪声


脉冲仿真器随时间演化噪声，因此其 Lindblad 规则使用速率或弛豫时间，而不会从有限概率推导速率。特别是，即使注册了 Lindblad 规则，[`PauliChannel`][fatqat.noise.PauliChannel] 仍是离散的 Simulator 通道：它的概率没有规定持续时间或换算约定。离散 Pauli 噪声请使用 [`Simulator`][fatqat.simulator.Simulator]，或者使用脉冲仿真器的 Lindblad 映射所支持的速率形式声明。

- [PulseOperation](pulse-operation.md)
- [PulseControl](pulse-control.md)
- [SampledWaveform](sampled-waveform.md)
- [门实现](gate-realization.md)
