---
title: 噪声
---

# 噪声


FatQat 将噪声与 [`Program`][fatqat.Program] 分离。把所需噪声源添加到 [`NoiseModel`][fatqat.NoiseModel]，再通过 `noise=...` 将该模型传给兼容的模拟器或仿真器。同一个程序可以复用于理想运行和含噪运行。

受控的理想—含噪对比参阅[以理想和含噪方式比较同一个 Program](../guide/ideal-and-noisy.md)。本节是选择器、支持情况、单位和验证规则的参考。

噪声类型位于 `fatqat.noise`。`NoiseModel` 也可以通过 `fatqat.NoiseModel` 使用。

## 选择噪声类型


概率描述匹配操作之后的一次模拟器通道应用。速率和弛豫时间描述在仿真器时间内作用的局部 Lindblad 算符。后端不会在这些形式之间自动转换。

**内置噪声类型**

| 噪声类型 | 接受的参数 | 作用对象 | 用户可见效果 |
| --- | --- | --- | --- |
| [`Depolarizing`][fatqat.noise.Depolarizing] | `p` 或 `rate`，必须且只能选一个 | `p`：选定操作数；`rate`：一个子系统 | 向最大混合态均匀混合 |
| [`PauliChannel`][fatqat.noise.PauliChannel] | Pauli 字符串概率映射或键值对序列 | 每个字符串字符对应一个量子比特 | `I`、`X`、`Y` 和 `Z` 字符串的随机混合 |
| [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] | `p` 或 `rate`，必须且只能选一个；每个相邻跃迁一个值 | 一个子系统 | 从能级 `k` 到 `k - 1` 的阶梯衰减 |
| [`PhaseDamping`][fatqat.noise.PhaseDamping] | `p`、`rate` 或 `t_phi`，必须且只能选一个 | 一个子系统 | 不发生布居转移的相干性衰减 |
| [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] | `t1` 和 `t2` | 一个子系统 | 能量弛豫和剩余纯退相干的组合 |
| [`Loss`][fatqat.noise.Loss] | 每个载体的概率 `p` | 匹配操作中的每个选定载体 | 在可感知占据状态的后端上永久移除载体 |
| [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | 列随机报告矩阵 | 每个被测子系统独立作用，或一个选定子系统 | 物理坍缩后对报告数字进行经典重采样 |

操作噪声与背景噪声、目标选择、组合、冲突和验证时机参阅[噪声模型](noise/model.md)。每个噪声类型页面说明其参数、单位和数学定义。

## 后端支持


支持情况取决于噪声形式、作用位置和后端。[后端支持](noise/backend-support.md#noise-backend-support)表格列出了开箱即用的功能、需要自定义实现映射的功能，以及某一后端类别完全无法支持的功能。

## 快速开始


下面的模型在每个 `CX` 之后添加一个联合通道，再对每次测量应用二元读出混淆：

```python
import numpy as np
import fatqat as fq
import fatqat.operations as ops

noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
noise.add(
    fq.noise.ReadoutConfusion(
        np.array([[0.98, 0.04], [0.02, 0.96]])
    )
)

backend = fq.simulator.Simulator(method="density_matrix", noise=noise)
```

对于脉冲后端，应使用模型的时间单位表示速率和弛豫时间。例如，参考超导量子比特模型使用 `"q0"` 等设备标签。下面的弛豫噪声会在该处的整个已用脉冲时间内持续作用：

```python
pulse_noise = fq.NoiseModel()
pulse_noise.add(
    fq.noise.ThermalRelaxation(t1=60_000.0, t2=80_000.0),
    targets="q0",
)
```

参考超导量子比特模型使用纳秒，中性原子模型使用微秒。请检查所选模型的 `time_unit`，不要根据数值大小猜测单位：参阅 [`time_unit`][fatqat.emulator.TransmonModel.time_unit]、[`time_unit`][fatqat.emulator.Atom2LevelModel.time_unit] 和 [`time_unit`][fatqat.emulator.Atom3LevelModel.time_unit]。

## API 页面


- [噪声模型](noise/model.md)
- [后端支持](noise/backend-support.md)
- [Depolarizing](noise/depolarizing.md)
- [PauliChannel](noise/pauli-channel.md)
- [AmplitudeDamping](noise/amplitude-damping.md)
- [PhaseDamping](noise/phase-damping.md)
- [ThermalRelaxation](noise/thermal-relaxation.md)
- [Loss](noise/loss.md)
- [ReadoutConfusion](noise/readout-confusion.md)
- [自定义噪声实现](noise/custom-implementations.md)
