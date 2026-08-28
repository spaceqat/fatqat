# 选择并运行中性原子工作流

FatQat 提供三个中性原子执行层次。请选择仍然包含所研究效应的最简层次：

<div class="grid cards" markdown>

- **门级阵列**

    [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 为有限量子比特门
    加入占用状态和不断变化的配对图。它适合快速检查加载、损失和连通性。

- **三个物理能级**

    [`Atom3LevelEmulator`][fatqat.emulator.Atom3LevelEmulator] 跟踪
    $\lvert 0\rangle$、$\lvert 1\rangle$ 和 $\lvert r\rangle$。它适合研究已校准门、
    选定位置的控制和 Rydberg 泄漏。

- **两个物理能级**

    [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] 跟踪
    $\lvert g\rangle$ 和 $\lvert r\rangle$。它适合研究全局驱动、失谐和多体
    Rydberg 动力学。

</div>

原子阵列模拟器就是[硬件配置模拟](hardware-profile-simulation.md)中介绍的配置；
它不会对哈密顿量积分。下面两个仿真器则使用[哈密顿量仿真](hamiltonian-emulation.md)
中的脉冲工作流。

## 只描述一次位置

两个物理仿真器都使用由 [`AtomArrangement`][fatqat.emulator.AtomArrangement]
描述的一组固定位置。默认情况下，Program 资源按声明顺序映射到这些位置：

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> arrangement = fq.emulator.AtomArrangement.chain(
...     num_sites=2,
...     spacing=6.0,
... )
>>> arrangement.num_sites
2
```

该排列是固定几何结构，而不是原子运输指令。它设定 Rydberg 相互作用使用的
距离，Program 必须为每个位置声明恰好一个维数为二的资源。在三能级仿真器中，
这些逻辑量子比特资源被嵌入物理量子三能级系统；这并不会把它们变成逻辑量子
三能级资源。

只有需要覆盖默认的声明顺序映射时，才使用
[`ResourceLayout`][fatqat.ResourceLayout]。

## 用三能级模型寻址一个位置

加载三能级参考模型，并运行一个普通的已校准旋转：

```pycon
>>> atom3_model = fq.emulator.Atom3LevelModel.from_document(
...     fq.emulator.load_model_document("atom3level.reference")
... )
>>> atom3 = fq.emulator.Atom3LevelEmulator(
...     atom3_model,
...     arrangement=arrangement,
... )
>>> calibrated = fq.Program(arrangement.num_sites)
>>> calibrated.add(ops.RX(np.pi / 2), 0)
>>> atom3_rho = atom3.run(calibrated).result().get_density_matrix()
>>> atom3_rho.shape
(9, 9)
```

`(9, 9)` 结果保留了两个位置的物理 `|r>` 布居。内置映射也会实现 `RY`、
`RZ` 和 `CZ`；当随包校准是实验起点时，请使用这一映射。

若要控制选定位置，请按位置寻址 Raman 或 Rydberg 通道。下面的直接控制块先
驱动位置 `0` 上的 Raman 跃迁，然后在操作进行到一半时启动 Rydberg 波形：

```pycon
>>> shaped = fq.emulator.SampledWaveform(
...     (0.0, 0.25, 0.5),
...     (0.0, 4.0, 0.0),
... )
>>> controls = (
...     fq.emulator.PulseControl(atom3_model.control.raman(0), shaped),
...     fq.emulator.PulseControl(
...         atom3_model.control.rydberg(0),
...         shaped,
...         start_offset=0.5,
...     ),
... )
>>> selected_site = fq.Program(arrangement.num_sites)
>>> selected_site.add(ops.PulseOperation(1.0, controls))
>>> selected_rho = atom3.run(selected_site).result().get_density_matrix()
>>> physical = np.real(np.diag(selected_rho)).reshape((3, 3), order="F")
>>> round(float(physical[2].sum()), 3)
0.146
```

这里最后一个值是位置 `0` 的物理 Rydberg 布居。当多个位置具有 Rydberg
布居时，几何结构也会产生相互作用，包括未在同一控制块中指名的位置。

## 用二能级模型驱动阵列

二能级模型使用全局驱动和失谐通道。它的默认门映射为空，因此 Program 通常
包含直接脉冲块：

```pycon
>>> atom2_model = fq.emulator.Atom2LevelModel.from_document(
...     fq.emulator.load_model_document("atom2level.reference")
... )
>>> atom2 = fq.emulator.Atom2LevelEmulator(
...     atom2_model,
...     arrangement=arrangement,
... )
>>> drive = fq.emulator.SampledWaveform(
...     (0.0, 0.5, 1.0),
...     (0.0, 0.5, 0.0),
... )
>>> detuning = fq.emulator.SampledWaveform(
...     (0.0, 0.5, 1.0),
...     (0.0, 0.1, 0.0),
... )
>>> global_controls = (
...     fq.emulator.PulseControl(atom2_model.control.drive(), drive),
...     fq.emulator.PulseControl(atom2_model.control.detuning(), detuning),
... )
>>> global_program = fq.Program(arrangement.num_sites)
>>> global_program.add(ops.PulseOperation(1.0, global_controls))
>>> atom2_state = atom2.run(global_program).result().get_statevector()
>>> atom2_state.shape
(4,)
>>> bool(np.isclose(np.linalg.norm(atom2_state), 1.0))
True
>>> round(float(1.0 - abs(atom2_state[0]) ** 2), 3)
0.054
```

两个控制都作用于每个位置。排列提供距离，模型提供带符号的相互作用强度，因此
改变间距可以在不改变 Program 的情况下改变动力学。默认情况下，每对位置都会
相互作用；`interaction_cutoff` 可以移除距离以外的项，但它是哈密顿量截断，
而不是阻塞半径。

对于这个短脉冲，最终约 5.4% 的概率位于全基态之外。该数值是波形、失谐、
间距、相互作用模型和持续时间共同产生的物理结果，而不是某个门标签的结果。

理想、未测量的运行会返回完整的二能级状态向量。加入支持的 Lindblad 噪声后，
它可以改为返回密度矩阵；末端测量则返回计数。线路中途测量、重置、条件和逐
位置直接控制不属于这套二能级工作流。

## 继续进行物理研究

接下来可学习 [PXP 复苏](../tutorials/pxp-z2-revival.md)、
[反铁磁链](../tutorials/antiferromagnetic-chain.md)或
[门级 GHZ 制备](../tutorials/atom-array-ghz8.md)教程。模型模式、单位、通道限制
和噪声支持请查阅[中性原子仿真器 API](../api/atom-emulators.md)。
