
# Simulator


[`Simulator`][fatqat.simulator.Simulator] 使用矩阵操作和有限噪声通道运行门级程序。它支持量子比特、量子多能级系统、混合寄存器维度和自定义实现映射。模拟器按程序原样运行，不会进行转译或路由。

## 快速开始


```python
import fatqat as fq
import fatqat.operations as ops

bell = fq.Program(2, 2)
bell.add(ops.H, 0)
bell.add(ops.CX, (0, 1))
bell.measure_all()

backend = fq.simulator.Simulator(method="statevector")
counts = backend.run(bell, shots=1000).result().get_counts()
```

默认实现映射覆盖 FATQAT 的内置矩阵门。状态方法还支持测量、重置和经典条件。`Barrier` 不产生数值作用。

## 方法


方法名称不区分大小写。`SV` 和 `DM` 是别名；只读 [`Simulator.method`][fatqat.simulator.Simulator.method] 属性返回完整名称。若程序的 Hilbert 空间维度为 `D`：

**模拟方法**

| 方法 | 结果 | 重置与有限通道 | 限制 |
| --- | --- | --- | --- |
| `statevector` / `SV` | `statevector`，形状 `(D,)` | 采样一条轨迹 | 随机最终状态表示一个 shot |
| `density_matrix` / `DM` | `density_matrix`，形状 `(D, D)` | 精确应用 | 比 `statevector` 使用更多内存 |
| `unitary` | `unitary`，形状 `(D, D)` | 拒绝 | 拒绝测量、条件、计数和 `initial_state` |
| `superop` | `superop`，形状 `(D**2, D**2)` | 精确应用 | 拒绝测量、条件、计数和 `initial_state` |

超算符使用列堆叠向量化：

```python
rho_out = (
    superop @ rho_in.reshape(-1, order="F")
).reshape(rho_in.shape, order="F")
```

对于无噪声程序，`superop` 等于 `numpy.kron(unitary.conj(), unitary)`。`n` 个量子比特的酉算符包含 `4**n` 个复数条目，超算符包含 `16**n` 个；只有在程序足够小、能够容纳结果时才使用算符方法。

## 运行时与执行


创建后端时选择 `runtime`。`"numba"` 是 [`Simulator`][fatqat.simulator.Simulator] 和超导量子比特配置的默认值；它在首次使用时编译内核，并支持多线程内核。`"numpy"` 直接运行而无需编译，是 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 的默认值。两个运行时都支持四种方法，但浮点结果或采样结果不一定逐比特相同。

`simulation_config` 会修改一次 [`Simulator.run`][fatqat.simulator.Simulator.run] 调用。其字符串值区分大小写。

**模拟控制**

| 键 | 默认值 | 接受的值与效果 |
| --- | --- | --- |
| `seed` | `None` | 使用非负 `int` 或 `None`；拒绝布尔值。控制测量、重置、通道、损失和读出采样。负值在执行开始时被拒绝，因此 [`fatqat.Job.result`][fatqat.Job.result] 会引发 `ValueError`。 |
| `shot_parallelism` | `"auto"` | `"auto"`、`"serial"`、`"threads"` 或 `"processes"`。显式并行模式要求运行只产生计数、逐 shot 执行且至少有两个 shot 和 worker。线程要求兼容的 Numba 状态向量运行。 |
| `kernel_parallelism` | `"auto"` | `"auto"`、`"serial"` 或 `"threads"`。线程要求 Numba，且不能与并行 shot 同时请求。 |
| `max_workers` | `None` | `None` 或正 `int`。限制所选并行模式；`1` 与显式并行请求冲突。 |
| `fusion` | `False` | 一个 `bool`。`True` 会融合兼容的相邻操作，Numba 的 `density_matrix`、`unitary` 和 `superop` 支持此功能。 |

自动选择至多使用一个并行轴。显式请求不受支持的选择会报错，而不会回退。只有当运行请求计数且必须独立演化各 shot 时，才有资格显式启用 shot 并行；例如程序包含中途测量、重置、条件或随机通道。只演化一次并仅对终端测量采样的电路不符合条件。线程 shot 要求 Numba 状态向量运行，且不支持原子占据状态生命周期；其他符合条件的工作负载可以使用进程。

只有在 Program、完整配置、FatQat 版本和执行环境都相同时，固定非负 `seed` 才能复现采样结果。改变运行时或并行执行模式可能改变随机数消耗。确定性结果不依赖种子，并遵循常规浮点容差。

实用基准测试工作流程参阅[性能与扩展](../guide/performance.md)；上表是规范配置契约。

## 自定义后端


构造函数还接受：

**后端选项**

| 参数 | 含义 |
| --- | --- |
| `implementation_map` | 操作的矩阵规则。`None` 使用 FATQAT 内置门集合。 |
| `noise` | 每次运行都使用的 [`NoiseModel`][fatqat.NoiseModel]。`None` 表示理想运行。 |
| `channel_implementation_map` | 将受支持通道描述符转换为有限通道的规则。`None` 使用 FATQAT 内置规则。 |

## 运行输入与结果


除 `simulation_config` 外，[`Simulator.run`][fatqat.simulator.Simulator.run] 还接受：

**运行输入**

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `shots` | `1024` | 用于计数或随机最终状态的采样次数。仅产生确定性状态或算符结果时不使用该值。 |
| `resource_layout` | `None` | 将每个程序量子引用分配给设备标签。通用模拟器按声明顺序使用整数标签。提供的布局必须完整、一一对应且与后端兼容。 |
| `initial_state` | `None` | 使每个 shot 从该状态而非全零状态开始。`statevector` 接受形状 `(D,)`；`density_matrix` 接受 `(D,)` 或 `(D, D)`。算符方法拒绝该参数。 |

只检查初态的形状。归一化以及密度矩阵的 Hermitian 性和正性由你负责。

`result_config` 有两个键。每个键都接受 `True`、`False` 或 `None`；省略或设为 `None` 时使用下表默认值。

**结果字段**

| 键 | 默认值 | 约束 |
| --- | --- | --- |
| `counts` | 程序包含测量时启用 | 要求整数 `shots > 0` |
| `final_state` | 方法原生状态或映射为确定性时启用 | 请求随机最终状态要求 `shots == 1` |

具体的最终状态字段名为 `statevector`、`density_matrix`、`unitary` 或 `superop`。读取可能未请求的字段前，请检查 `fatqat.Result.available_data`。

`run()` 返回立即完成的 [`Job`][fatqat.Job]。程序和选项验证错误通常直接抛出。执行或结果组装期间的错误会存入作业，并由 [`fatqat.Job.result`][fatqat.Job.result] 再次抛出。结果访问器和计数顺序的直观说明参阅[从一次运行中获取答案](../guide/interpret-results.md)。精确状态轴元数据在 [Result](result.md) 中规定。

## 噪声


矩阵模拟没有物理时间线。因此，内置阻尼和退极化描述符使用其概率形式，并在操作边界处应用。速率形式、背景源和 [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] 会被拒绝；应先用 `as_channels(duration)` 转换热弛豫。自定义描述符需要匹配通道规则。[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 还支持原子损失。

[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] 无需运行程序即可验证模型。某种方法仍可能施加更严格规则：例如，后端可能识别某有限通道，但若该通道与程序匹配，`unitary` 会拒绝。选择器和支持表参阅[噪声](noise.md)。

## 扫描


[`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep] 绑定完整对象键参数批次的每一行，并返回一个包含有序 `list[Result]` 的立即完成作业。批次和行验证错误直接抛出；执行故障会产生失败的扫描作业，且不返回部分结果列表。每一行会复用给定种子，因此采样误差可能相关。引导式扫描参阅[模拟量子程序](../guide/simulation.md)。接受的批次形状在上文中规定。

## API


::: fatqat.simulator.Simulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
