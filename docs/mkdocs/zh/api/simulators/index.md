---
title: 模拟器
---

# 模拟器


本节将门级、矩阵型的 `Simulator` 后端统一称为“模拟器”。FatQat 提供一个通用线路模拟器和三种硬件配置，它们使用相同的运行与结果 API。
若要进行不受限制的门级工作，请选择 [`Simulator`][fatqat.simulator.Simulator]；
若程序必须遵循原生门集合、布局或连通性规则，请选择硬件配置。

请先阅读[选择物理建模的细致程度](../../guide/execution-models.md)来选择执行层次，
或按照[在硬件配置上测试 Program](../../guide/hardware-profile-simulation.md)中的
步骤使用硬件配置。

超导硬件配置使用固定的矩形网格。
[`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] 接受 IBM 风格
的原生门和最近邻 `CZ`。
[`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] 接受
原生旋转以及最近邻 `iSwap` 和 `CZ`。二者都提供可选的参考噪声模型。

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 没有固定连通性。
`Pair` 和 `Unpair` 会改变哪些原子能够相互作用，`Put` 和 `Loss` 则控制占用。

这些配置会校验原样写出的程序：不会转译或路由程序，也不会复现某一具名处理器。
当时序或哈密顿量演化很重要时，请使用[脉冲仿真器](../emulators/index.md)。

**选择模拟器**

| 类 | 适用场景 | 主要约束 |
| --- | --- | --- |
| [`Simulator`][fatqat.simulator.Simulator] | 通用线路仿真与自定义矩阵实现 | 无设备拓扑 |
| [`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] | IBM 风格原生门与网格实验 | `X`、`SX`、`RZ`；最近邻 `CZ` |
| [`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] | Google 风格原生门与网格实验 | `RX`、`RY`、`RZ`；最近邻 `iSwap` 和 `CZ` |
| [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] | 中性原子占用、损失与动态连通性 | `RX`、`RY`、`RZ` 和已配对的 `CZ` |

- [Simulator](../simulator.md)
- [SCQubitIBMSimulator](sc-qubit-ibm.md)
- [SCQubitGoogleSimulator](sc-qubit-google.md)
- [AtomArraySimulator](atom-array.md)
