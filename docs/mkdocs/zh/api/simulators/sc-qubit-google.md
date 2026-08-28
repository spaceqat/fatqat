<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# SCQubitGoogleSimulator


[`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] 将
[`Simulator`][fatqat.simulator.Simulator] 执行模型应用到可配置的 Google 风格
超导网格。可用它测试原生旋转门以及最近邻 `iSwap`/`CZ` 程序。它不是某台
具名 Google 处理器的模型，不会转译、路由或调度程序，也不会复现硬件校准数据。

**硬件配置**

| 属性 | 值 |
| --- | --- |
| 默认设备 | `grid_size=(4, 4)`；16 个按行优先排列的量子比特 |
| 全局原生门 | [`fatqat.operations.RX`][fatqat.operations.RX]、[`fatqat.operations.RY`][fatqat.operations.RY]、[`fatqat.operations.RZ`][fatqat.operations.RZ] |
| 连通原生门 | 水平或垂直相邻位置上的 [`fatqat.operations.iSwap`][fatqat.operations.iSwap] 和 [`fatqat.operations.CZ`][fatqat.operations.CZ] |
| 其他内置操作 | 测量与 [`fatqat.operations.Reset`][fatqat.operations.Reset] 遵循 [`Simulator`][fatqat.simulator.Simulator] 所述的方法规则 |
| 方法 | [`Simulator`][fatqat.simulator.Simulator] 支持的全部方法；默认为 `statevector` |
| 运行时 | 默认为 `numba`；也支持 `numpy` |
| 噪声 | 默认为理想情况；可显式获取内置参考模型 |

## 原生门与布局


设备标签按行优先排列。在默认网格上，标签为：

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

每条网格边的两种操作数顺序都合法。因此，`iSwap` 可用于设备标签 `(0, 1)`
和 `(1, 0)`，但不可用于 `(0, 5)`。`CZ` 对于任意正的
`grid_size=(rows, columns)` 都遵循同一规则。

使用自动布局时，普通程序的量子比特按声明顺序映射到设备标签 `0, 1, ...`。
单个 [`GridRegister`][fatqat.GridRegister] 则映射到设备左上角，同时保留行列
坐标。在这种自动模式下，它必须是程序唯一的量子寄存器，而且两个设备轴都必须
容得下它。显式而完整的 [`ResourceLayout`][fatqat.ResourceLayout] 可以采用
不同方式放置程序引用。容量与仅限量子比特的约束仍然有效。

后端不会分解非原生操作。尤其是，[`fatqat.operations.CX`][fatqat.operations.CX]
和 [`fatqat.operations.SX`][fatqat.operations.SX] 会被拒绝。请检查
[`SCQubitGoogleSimulator.implementation_map`][fatqat.simulator.SCQubitGoogleSimulator.implementation_map]
以查看门集合和合法操作数元组：

```python
import fatqat as fq
import fatqat.operations as ops

backend = fq.simulator.SCQubitGoogleSimulator(grid_size=(2, 3))
native = backend.implementation_map

assert native.supports(ops.RY)
assert native.supports(ops.iSwap, device_operands=(1, 4))
assert not native.supports(ops.iSwap, device_operands=(0, 4))
```

对于随处可用的门，`device_operands_for(operation)` 返回空集合；对于受连通性
限制的门，则返回有序元组。

## 内置噪声


如果没有提供噪声模型，模拟器会保持理想。若要使用内置配置，请显式请求：

```python
import fatqat as fq

Sim = fq.simulator.SCQubitGoogleSimulator
backend = Sim(noise=Sim.default_noise_model())
```

该配置使用 `T1 = 60 us` 和 `T2 = 48 us`。每次调用都会返回一个新的
[`NoiseModel`][fatqat.NoiseModel]，传给模拟器前可以继续扩展它。

**内置配置**

| 操作 | 持续时间 | 噪声 |
| --- | --- | --- |
| `RX`、`RY`、`RZ` | 20 ns | 由 T1/T2 推导的振幅阻尼与相位阻尼 |
| `iSwap` | 30 ns | 每个量子比特上的弛豫，随后是 `p = 0.001` 的联合退极化噪声 |
| `CZ` | 50 ns | 每个量子比特上的弛豫，随后是 `p = 0.001` 的联合退极化噪声 |
| 测量 | — | `P(report 1 \| true 0) = 0.02` 和 `P(report 0 \| true 1) = 0.04` |

与 IBM 风格配置不同，在此配置中，`RZ` 是一个有噪声、持续 20 ns 的物理
旋转。不同方法如何执行通道请参阅[噪声](../noise.md)。

## API


[`Simulator.method`][fatqat.simulator.Simulator.method]、
[`SCQubitGoogleSimulator.implementation_map`][fatqat.simulator.SCQubitGoogleSimulator.implementation_map]、
[`Simulator.run`][fatqat.simulator.Simulator.run]、
[`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep] 和
[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model]
遵循通用 Simulator API，并在下方列出，以提供完整的类参考。

::: fatqat.simulator.SCQubitGoogleSimulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
