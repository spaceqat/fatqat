<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# AtomArraySimulator


[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 在
[`Simulator`][fatqat.simulator.Simulator] 执行模型上加入中性原子占用、损失
和动态配对。可用它检查程序是否符合这些约束。它不是哈密顿量或运输模型；当
脉冲时序和物理相互作用很重要时，请使用[中性原子仿真器](../atom-emulators.md)。

**硬件配置**

| 属性 | 值 |
| --- | --- |
| 容量 | 默认不受限制；`num_sites` 可设置一个正的固定上限 |
| 原生门 | [`fatqat.operations.RX`][fatqat.operations.RX]、[`fatqat.operations.RY`][fatqat.operations.RY]、[`fatqat.operations.RZ`][fatqat.operations.RZ] 和 [`fatqat.operations.CZ`][fatqat.operations.CZ] |
| 连通性 | 无固定拓扑；只有两个原子处于配对状态时才能合法执行 `CZ` |
| 维数 | 仅限量子比特 |
| 方法 | [`Simulator`][fatqat.simulator.Simulator] 的全部方法适用于不包含 `Put` 或损失的程序；原子生命周期要求 `statevector` 或 `density_matrix` |
| 运行时 | 默认为 `numpy`；也支持 `numba` |
| 噪声 | 默认为理想情况；无内置参考模型 |

## 容量、映射与原生操作


`num_sites=None` 不设置容量限制。正值会拒绝声明的量子子系统数量超过可用
位置的程序。寄存器按声明顺序映射到整数设备标签；[`GridRegister`][fatqat.GridRegister]
会被扁平化，其坐标在此后端上没有物理含义。

原生操作为 `RX`、`RY`、`RZ` 和 `CZ`。模拟器不会分解其他门，因此即使
原子已经配对，`CX` 也会被拒绝。
[`AtomArraySimulator.implementation_map`][fatqat.simulator.AtomArraySimulator.implementation_map]
列出原生门集合；程序当前的 `Pair` 状态决定某个特定 `CZ` 是否允许执行。

配对关系会随程序运行而改变。无条件的
[`fatqat.operations.Pair`][fatqat.operations.Pair] 会连接两个位置，
[`fatqat.operations.Unpair`][fatqat.operations.Unpair] 则断开二者。对未配对
的一对位置执行 `CZ` 会被拒绝。配对操作不会改变量子态，但附加到它们的噪声
仍会生效。不支持带条件的 `Pair` 和 `Unpair`。

## 占用与损失


每次采样都会单独追踪占用状态。其初始值取决于程序是否使用原子生命周期：

**占用规则**

| 程序 | 初始占用与行为 |
| --- | --- |
| 不包含 `Put`，也没有匹配的 [`fatqat.noise.Loss`][fatqat.noise.Loss] 源 | 每个已声明位置均有原子。除原生门和配对规则外，程序行为与通用模拟器相同。 |
| 包含 `Put`，或某个操作匹配 [`fatqat.noise.Loss`][fatqat.noise.Loss] 源，即使 `p=0` | 每个位置初始为空。[`fatqat.operations.Put`][fatqat.operations.Put] 会在目标位置加载一个新的 `\|0>` 原子。 |

在已占用位置执行 `Put` 不会产生任何效果。对于某次采样，在空位置或先前已损失
的位置执行门或重置同样无效。测量仍会报告擦除，配对仍会改变连通性，之后的
`Put` 也可以重新填充已损失的位置。

[`fatqat.noise.Loss`][fatqat.noise.Loss] 可以移除门的目标、使 `Put` 失败，
或模拟 `Pair` 和 `Unpair` 期间的损失。只有选择器匹配某个操作时，它才会影响
运行；即使 `p=0`，匹配的源仍会启用显式占用。这是唯一接受 `Loss` 的门级
模拟器。缺失原子会使原本有效的已配对 `CZ` 在该次采样中不产生效果；未配对的
`CZ` 则会在执行前被拒绝。

测量空位置会报告擦除数字 `2`。擦除会绕过读出混淆噪声，因为不存在可读取的
已占用量子比特。原子损失会使最终状态具有随机性，因此 `final_state=True`
要求 `shots == 1`。带测量且含损失的运行默认返回计数，但不返回任意一条轨迹
的最终状态。

## 示例


```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.Put, (0, 1))
program.add(ops.Pair, (0, 1))
program.add(ops.RY(0.4), 0)
program.add(ops.CZ, (0, 1))
program.measure_all()

backend = fq.simulator.AtomArraySimulator(num_sites=2)
counts = backend.run(program, shots=1000).result().get_counts()
```

原子生命周期无法用 `unitary` 或 `superop` 表示，因为占用是量子矩阵之外的
状态。只有配对时可以使用算符方法：它会改变哪些原生双量子比特操作合法，但
不会创建占用状态。

## API


[`Simulator.method`][fatqat.simulator.Simulator.method]、
[`AtomArraySimulator.implementation_map`][fatqat.simulator.AtomArraySimulator.implementation_map]、
[`Simulator.run`][fatqat.simulator.Simulator.run]、
[`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep] 和
[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model]
遵循通用 Simulator API，并在下方列出，以提供完整的类参考。

::: fatqat.simulator.AtomArraySimulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
