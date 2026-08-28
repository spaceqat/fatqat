# 让 Program 进入物理动力学

FatQat 仿真器仍然接受 [`Program`][fatqat.Program]，但它改变了“执行”的含义。
它不再应用离散门变换，而是构建物理调度并对含时哈密顿量进行积分；如有请求，
也会计算 Lindblad 演化。

两种编程路径汇入同一个调度：

```text
Program gate ----> calibration ----\
                                  +----> physical schedule
PulseOperation --> controls ------/               |
                                                   v
                                    Hamiltonian/Lindblad evolution
```

模型提供物理能级、单位、漂移、通道和耦合。门实现映射会把普通门转换为已校准的
控制；直接脉冲则显式提供这些控制。后续页面会将这套共同工作流分别用于
Transmon 和原子。

## 运行已校准的门

先使用随包提供的物理模型作为可复现基线，再构建对应仿真器。这里，一个单量子
比特 Program 会在包含两个三能级 Transmon 的模型上实现：

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model = fq.emulator.TransmonModel.from_document(
...     fq.emulator.load_model_document("transmon.reference")
... )
>>> backend = fq.emulator.TransmonEmulator(model)
>>> calibrated_program = fq.Program(1)
>>> calibrated_program.add(ops.RX(np.pi / 2), 0)
>>> calibrated_result = backend.run(calibrated_program).result()
>>> calibrated_result.get_density_matrix().shape
(9, 9)
```

`RX` 仍是普通的 Program 操作。执行时，仿真器的门映射从校准中取得脉冲方案，
并将其绑定到模型的驱动通道。物理模型包含两个量子三能级系统，因此返回矩阵
覆盖全部 $3^2$ 个基态，也包括未被寻址的第二个 Transmon。这个物理量子
三能级状态并不是逻辑量子三能级 Program。

随包模型和校准都是参考快照。当物理系统或校准发生变化时，请提供经过验证的
模型文档和实现映射；不要把随包数据理解为实时设备校准。

## 直接添加脉冲

直接脉冲包含三个部分：波形样本、它们驱动的物理通道，以及控制块的持续时间。
下面的代码依次将三者构建为 [`SampledWaveform`][fatqat.emulator.SampledWaveform]、
[`PulseControl`][fatqat.emulator.PulseControl]，最后是
[`PulseOperation`][fatqat.operations.PulseOperation]：

```pycon
>>> duration = 20.0
>>> waveform = fq.emulator.SampledWaveform(
...     (0.0, 10.0, duration),
...     (0.0, 0.02, 0.0),
... )
>>> control = fq.emulator.PulseControl(
...     model.control.drive("q0"),
...     waveform,
... )
>>> pulse = ops.PulseOperation(duration=duration, controls=(control,))
>>> direct_program = fq.Program(1)
>>> direct_program.add(pulse)
>>> direct_result = backend.run(direct_program).result()
>>> direct_result.get_density_matrix().shape
(9, 9)
```

请注意，`direct_program.add(pulse)` 没有逻辑目标。通道已经指定物理 Transmon
`q0`，[`ResourceLayout`][fatqat.ResourceLayout] 不会重新映射该地址。模型
与通道定义时间和值的单位；API 参考记录了每个仿真器的准确取值域。

## 将脉冲理解为连续演化

上面的两个波形样本并非两个门步骤。在完整的 20 单位区间内，仿真器会对驱动
进行插值，将其与漂移和耦合项组合，再对物理状态积分。因此，结果会保留模型中
的每个能级，包括逻辑 Program 未声明的能级。

下一章会把这一机制用于具体的 Transmon 实验：比较已校准旋转和直接驱动，
并让由此产生的泄漏清晰可见。

## 理解调度

同一个 `PulseOperation` 内的所有控制共享其时间区间；`start_offset` 可以
在该区间内移动单个波形。不同操作之间，轻量调度器会为占用相同物理资源的控制
块保留源码顺序，也可以让彼此独立的控制块重叠。当这种布局很重要时，请选择
`"ASAP"` 或 `"ALAP"`，并通过 `simulation_config` 传入。

漂移和后台连续噪声会在所有已经过的时间中持续演化。对于支持经典条件的仿真器，
为假的条件可以跳过控制块，但不会删除其持续时间，因此模型仍会在该区间内演化。
正因如此，哈密顿量仿真器回答的问题不同于硬件配置模拟器。

继续学习 [Transmon 仿真](transmon-emulation.md)或
[中性原子仿真](neutral-atom-emulation.md)。脉冲对象、调度、求解器与模型契约
请查阅[脉冲控制 API](../api/pulse-control/index.md)和
[仿真器 API](../api/emulators/index.md)。
