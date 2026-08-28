# 选择物理建模的细致程度

后端不仅是程序的提交位置，它还决定 FatQat 如何理解“运行这个程序”。通用
模拟器遵循逻辑线路演化；硬件配置会加入设备规则；仿真器则按时间追踪物理模型。

保持 `Program` 不变时，三者的区别最为直观。下面的单量子比特旋转可被三个
示例共同理解：

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> program = fq.Program(1)
>>> program.add(ops.RX(np.pi / 2), 0)
```

=== "通用模拟器"

    通用 [`Simulator`][fatqat.simulator.Simulator] 直接应用逻辑门操作，不会把
    量子比特分配到某一具体设备：

    ```pycon
    >>> general = fq.simulator.Simulator(method="statevector", runtime="numpy")
    >>> general_result = general.run(
    ...     program,
    ...     shots=0,
    ...     result_config={"final_state": True},
    ... ).result()
    >>> np.round(np.abs(general_result.get_statevector()) ** 2, 3)
    array([0.5, 0.5])
    ```

    这是回答算法问题最快捷的方式：旋转后得到 `0` 和 `1` 的概率相等。

=== "硬件配置"

    硬件配置模拟器仍在线路层次演化量子门，但会先检查程序是否由原生操作组成、
    是否能够进行物理布局。此处使用的 Google 风格配置将 `RX` 视为原生门：

    ```pycon
    >>> profile = fq.simulator.SCQubitGoogleSimulator(
    ...     grid_size=(1, 1),
    ...     runtime="numpy",
    ... )
    >>> profile_result = profile.run(
    ...     program,
    ...     shots=0,
    ...     result_config={"final_state": True},
    ... ).result()
    >>> np.round(np.abs(profile_result.get_statevector()) ** 2, 3)
    array([0.5, 0.5])
    ```

    数值答案相同，但结论更强：这条特定指令也属于所选原生门集合，并符合其
    资源模型。

=== "物理仿真器"

    Transmon 仿真器通过随包提供的脉冲校准实现 `RX`，并对三能级物理模型进行
    积分：

    ```pycon
    >>> model = fq.emulator.TransmonModel.from_document(
    ...     fq.emulator.load_model_document("transmon.reference")
    ... )
    >>> emulator = fq.emulator.TransmonEmulator(model)
    >>> physical_result = emulator.run(program, shots=0).result()
    >>> physical_state = physical_result.get_density_matrix()
    >>> physical_state.shape
    (9, 9)
    >>> round(float(np.trace(physical_state).real), 12)
    1.0
    ```

    参考模型包含两个物理三能级 Transmon，因此即使 `Program` 只寻址一个
    量子比特，其状态也比仅有两个振幅的逻辑状态更大。多出的空间用于表示未被
    寻址的硬件和泄漏。

## 每个层次能够回答什么

| 想要了解的问题 | 建议从这里开始 | FatQat 建模的内容 |
|---|---|---|
| 算法能否产生预期的逻辑行为 | 通用模拟器 | 逻辑子系统上的线路操作 |
| 编写的程序是否符合目标设备的原生操作和资源规则 | 硬件配置模拟器 | 线路演化，以及布局、连通性、占用和可选的配置噪声 |
| 量子门或控制如何表现为有时序的物理动力学 | 仿真器 | 能级、漂移、耦合、脉冲、哈密顿量和兼容的开放系统噪声 |

请从包含所需效应的最简模型开始。只有当设备规则或连续时间物理会影响结果时，
才加入这些细节。

## 不同层次会进行不同的能力检查

三条路径都接受 `Program`、返回 `Job`，并通过 `Result` 暴露数据，但接受的
指令集合并不相同。

例如，通用模拟器支持逻辑量子多能级系统，以及具有混合局域维数的寄存器。
当前的硬件配置模拟器和脉冲仿真器只接受维数为二的逻辑资源。Transmon
仿真器可能返回物理量子三能级状态，因为它保留了一个泄漏能级；这与在 Program
中声明逻辑量子三能级系统不同。

硬件配置会校验原样写出的 Program，而不会对其进行转译或路由。逻辑标签与
设备标签不同时，[`ResourceLayout`][fatqat.ResourceLayout] 会明确二者的
绑定关系。[测试硬件配置](hardware-profile-simulation.md)一章将演示一次失败的
布局及其修正过程。

接下来，可以阅读[模拟量子程序](simulation.md)，在线路层次探索状态、扫描和
输出；如果重点已经是时间与控制，也可直接进入[哈密顿量级仿真](hamiltonian-emulation.md)。
