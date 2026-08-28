<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 超导量子比特脉冲仿真器


[`TransmonEmulator`][fatqat.emulator.TransmonEmulator] 在三能级超导量子比特模型上运行 [`Program`][fatqat.Program]。它在完整物理模型上演化采样控制，`run()` 返回立即完成的 [`Job`][fatqat.Job]。

校准门和直接驱动的分步指南参阅[仿真超导系统](../guide/transmon-emulation.md)。

普通门使用 `gate_implementation_map`。[`PulseOperation`][fatqat.operations.PulseOperation] 包含自身物理通道，不使用该映射。直接编写脉冲参阅[脉冲控制](pulse-control/index.md)。

除非显式写出导入路径，本页受支持的导入均来自 `fatqat.emulator`。`Transmon`、`Coupling` 和 GHz 转换辅助函数从 `fatqat.emulator.superconducting` 导入。

## 创建仿真器


加载打包模型并构造仿真器：

```python
import json
import fatqat as fq

model_document = fq.emulator.load_model_document("transmon.reference")
model = fq.emulator.TransmonModel.from_document(model_document)
backend = fq.emulator.TransmonEmulator(model)
```

若要使用显式校准，请在构造仿真器前根据校准构建门映射：

```python
with open("calibration.json", encoding="utf-8") as stream:
    calibration = fq.emulator.TransmonCalibration(json.load(stream))
gate_map = fq.emulator.default_transmon_gate_implementation_map(
    model=model,
    calibration=calibration,
)
backend = fq.emulator.TransmonEmulator(
    model,
    gate_implementation_map=gate_map,
)
```

打包校准是用于模拟的参考配置，不是硬件校准。若要自定义，应提供完整校准文档，而不是局部补丁。

默认情况下，程序量子比特按声明顺序绑定到 `model.subsystem_ids`。`run(resource_layout=...)` 和 `propagator(resource_layout=...)` 接受显式 [`ResourceLayout`][fatqat.ResourceLayout]，其中设备标签是模型子系统 ID。未被寻址的模型超导量子比特仍参与完整物理状态，因此结果和传播子维度中仍会包含其三维因子。它们有序的公共身份会出现在结果的 `state_axes` 元数据中。

`TransmonEmulator(...)` 接受以下可选参数：

**构造函数选项**

| 参数 | 含义 |
| --- | --- |
| `noise` | 一个 [`NoiseModel`][fatqat.NoiseModel]。`None` 表示无噪声。 |
| `lindblad_implementation_map` | 将通道描述符映射到局部坍缩算符的 [`LindbladImplementationMap`][fatqat.noise.LindbladImplementationMap]。`None` 使用 [`default_lindblad_implementation_map`][fatqat.noise.default_lindblad_implementation_map]。内置覆盖范围参阅[脉冲仿真器](noise/backend-support.md#noise-emulator-support)。 |
| `gate_implementation_map` | 将操作类别和设备标签映射到脉冲定义的 [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap]。`None` 使用内置映射。 |

## 运行


[`run`][fatqat.emulator.TransmonEmulator.run] 接受以下 `simulation_config` 键：

**`simulation_config` 键**

| 键 | 类型 | 默认值 | 效果与约束 |
| --- | --- | --- | --- |
| `seed` | `int` 或 `None`；不能是 `bool` | `None` | 为测量和读出采样设定种子。使用非负整数；`None` 使用新的熵。 |
| `schedule_mode` | `"ASAP"` 或 `"ALAP"` | `"ASAP"` | 在保持依赖关系和物理资源冲突约束的前提下，尽早或尽晚放置操作。 |

这是仅有的两个键。脉冲仿真器拒绝矩阵后端的 `shot_parallelism`、`kernel_parallelism`、`max_workers` 和 `fusion` 设置。

**`result_config` 键**

| 键 | 类型 | 默认值 | 效果与约束 |
| --- | --- | --- | --- |
| `counts` | `bool` 或 `None` | `None` | `True` 请求采样经典计数，`False` 禁用，`None` 在存在测量时启用。计数要求 `shots` 为正整数。 |
| `final_state` | `bool` 或 `None` | `None` | `True` 请求完整模型的终端物理密度矩阵，`False` 禁用，`None` 在没有测量时启用。存在测量时要求 `shots == 1`。 |

两个配置参数都必须是 `dict` 或 `None`；未知键会被拒绝。

每次运行都从每个超导量子比特处于物理 `|0>` 的直积态开始。脉冲仿真器不接受 `initial_state` 参数。

对于模型中的全部 `m` 个超导量子比特，密度矩阵形状为 `(3**m, 3**m)`。测量先采样物理能级，将 `0, 1, 2` 映射到 `0, 1, 1`，再应用任何经典读出混淆矩阵。重置会制备物理 `|0>`。

结果元数据包含生效的运行和结果设置，但不包含模型或校准文档。

`run()` 会在返回作业前抛出验证错误。如果返回作业后执行失败，`job.result()` 会引发 [`BackendExecutionError`][fatqat.errors.BackendExecutionError]。

## 传播子


[`propagator`][fatqat.emulator.TransmonEmulator.propagator] 为完整物理模型返回一个复 NumPy 数组。测量、重置和经典条件因无法定义单一相干算符而被拒绝。在非零持续时间演化中应用 Lindblad 噪声的程序也会被拒绝。没有时间经过时，基于速率的噪声不产生作用。

中间虚拟参考系更新始终会旋转后续对相位敏感的控制。`apply_final_frame=True`（默认）还会合成剩余的终端虚拟参考系变换；`False` 只省略最后这一变换。
结果使用每个子系统的近共振旋转参考系，与传统量子比特 `RZ` 可能相差全局相位；比较理想矩阵时应忽略相位。

## 参考


::: fatqat.emulator.TransmonEmulator
    options:
      members:
        - "run"
        - "propagator"
        - "validate_noise_model"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

## 物理模型与校准


`TransmonModel.from_document(...)` 接受已解码、兼容 JSON 的模型映射；禁止直接构造模型。校准构造函数另行接受其已解码校准映射。打包参考使用 `load_model_document("transmon.reference")`；自定义文档使用 `json.load` 或其他 JSON 读取器。文档必须匹配所选的 `format` ID 和版本。缺少或未知键、不受支持的版本、非有限值及超出规定 JSON 兼容类型的值都会被拒绝。

控制和参考系地址指向模型资源。调用 `run()` 或 `propagator()` 时会报告无效地址。

内置模型包含固定量子三能级超导量子比特及任意无向耦合图。耦合声明可驱动受控交换操作的位置；它不是残余的常开交换 Hamiltonian。频率定义隐式共振载波。

### 模型文档


::: fatqat.emulator.FormatIdentity
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.ModelIdentity
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.CalibrationIdentity
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.available_model_documents

::: fatqat.emulator.load_model_document

### 类 `fatqat.emulator.TransmonModel` { #fatqat.emulator.TransmonModel }

使用 [`from_document`][fatqat.emulator.TransmonModel.from_document] 创建实例；不支持直接构造。

::: fatqat.emulator.TransmonModel.from_document

::: fatqat.emulator.TransmonCalibration
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.superconducting.Transmon
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.superconducting.Coupling
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.default_transmon_calibration

`model.format` 标识文档模式；`model.kind` 和 `model.identity` 标识模型类别和快照。`calibration.format` 和 `calibration.identity` 标识校准文档，后者没有目标模型字段。

**`model.subsystems`**

按顺序排列的超导量子比特记录，包含 `id`、`frequency_ghz` 和 `anharmonicity_ghz`。

**`model.couplings`**

无向边记录，包含 `id` 和两个 `subsystem_ids`。

**`model.annihilation`、`model.number`**

只读局部量子三能级矩阵。它们是局部算符，而不是完整模型张量展开。升算符不单独存储，可通过 `model.annihilation.conj().T` 推导。

### 单位


两种单位支配规则发出的每个脉冲，且都来自模型：

**`model.time_unit`（`"ns"`）**

`PulseDefinition.duration` 以及每个 `PulseControl` 波形和 `start_offset` 的坐标单位。

**`model.control_unit`（`"rad/ns"`）**

三种通道的每个 `SampledWaveform.values` 条目的单位。这是**角**速率，不是普通频率。

模型和校准文档以 GHz 存储普通频率。脉冲波形使用角速率，因此自定义门规则应使用 [`fatqat.emulator.superconducting.angular_rate_from_ghz`][fatqat.emulator.superconducting.angular_rate_from_ghz] 转换文档值。

::: fatqat.emulator.superconducting.angular_rate_from_ghz

::: fatqat.emulator.TransmonModel.time_unit

::: fatqat.emulator.TransmonModel.control_unit

::: fatqat.emulator.TransmonModel.subsystem_ids

::: fatqat.emulator.TransmonModel.physical_dimension

`model.control` 命名空间选择 Hamiltonian 机制，`frame` 选择虚拟驱动相位。其方法也可按名称从 `model.available_controls` 映射中取得。每个映射条目描述一种受支持控制类别，而不是所有已完全绑定的通道实例。选择器公开 `scope`、必需的 `operands`、`coefficient_domain` 和 `coefficient_unit`，便于轻量检查。调用选择器会返回通道地址。运行程序时，仿真器会检查资源名称、已声明配对、波形类型和值。

```python
drive = model.control.drive("q0")
detuning = model.control.detuning("q1")
exchange = model.control.exchange("q0", "q1")

assert model.available_controls["drive"] is model.control.drive

for name, selector in model.available_controls.items():
    print(name, selector.scope, selector.operands,
          selector.coefficient_domain, selector.coefficient_unit)
```

::: fatqat.emulator.TransmonModel.control

::: fatqat.emulator.TransmonModel.available_controls

::: fatqat.emulator.TransmonModel.frame

### 校准配方


内置校准模式包含 `rx_ry`、`iswap` 和逐边 `cz` 配方。[`RZ`][fatqat.operations.RZ] 是虚拟门，没有校准配方。

公共标量单位访问器 `recipe_time_unit`、`recipe_frequency_unit` 和 `recipe_dimensionless_unit` 描述存储的配方量。它们不同于模型的脉冲坐标 `time_unit` 和 `control_unit`。

## 脉冲实现映射


[`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] 实现普通门。超导量子比特构造函数把此能力命名为 `gate_implementation_map`；直接控制会绕过它。标准构建器返回一个新映射，其中包含针对一个模型和校准的内置 `RX`、`RY`、`RZ`、`iSwap` 和 `CZ` 规则。

::: fatqat.emulator.default_transmon_gate_implementation_map

接受的规则形式和错误参阅[门实现](pulse-control/gate-realization.md)。

## 直接控制


同一模型通道可以在没有门实现规则时直接使用。驱动和失谐解析一个已声明超导量子比特；交换解析两个超导量子比特及其已声明耦合。驱动接受表示两个正交分量的复数包络，失谐和交换要求实数值。脉冲时间使用上述模型单位。当前超导量子比特模型除要求有限值外，不施加振幅或持续时间限制。

构造与时序参阅 [PulseOperation](pulse-control/pulse-operation.md)、[PulseControl](pulse-control/pulse-control.md) 和 [SampledWaveform](pulse-control/sampled-waveform.md)。`iSwap` 是内置实现使用交换的门；`iSwap` 不是通道名称。

## Lindblad 噪声与自定义规则


传入 [`LindbladImplementationMap`][fatqat.noise.LindbladImplementationMap] 可以添加或替换 Lindblad 噪声规则。映射 API 参阅[自定义噪声实现](noise/custom-implementations.md)，内置后端支持表参阅[脉冲仿真器](noise/backend-support.md#noise-emulator-support)。省略映射时，默认注册 [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping]、[`PhaseDamping`][fatqat.noise.PhaseDamping] 和 [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation]；显式映射会替换这些规则。量子三能级振幅阻尼需要两个相邻能级速率。速率使用每纳秒，`t1`、`t2` 和 `t_phi` 使用纳秒。接受背景和普通操作域生成元。有限概率形式、`Loss` 和非局部声明会被拒绝。

概率形式通道不会转换为速率。特别是，即使注册了规则，[`PauliChannel`][fatqat.noise.PauliChannel] 仍仅由 Simulator 支持。参阅[连续时间噪声](pulse-control/index.md#pulse-probability-noise)。速率形式的 `Depolarizing` 声明也要求 Lindblad 映射注册该类型；超导量子比特默认映射未注册。

运行程序前调用 [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] 验证噪声模型。程序专用选择器会在运行时检查。

## 中性原子脉冲仿真器


三能级和二能级原子后端也接受可选门实现映射和 Lindblad 实现映射。`Atom3LevelEmulator` 具有内置门配方和逐位点直接控制。`Atom2LevelEmulator` 的内置门映射为空，并具有全局直接控制；用户提供的映射可以添加门规则。其 API 参阅[中性原子仿真器](atom-emulators.md)，两种后端的选择指南参阅[选择并运行中性原子工作流程](../guide/neutral-atom-emulation.md)。
