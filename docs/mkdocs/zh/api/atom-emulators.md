<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 中性原子仿真器


两个中性原子仿真器都遵循标准 [`Simulator`][fatqat.simulator.Simulator] 工作流程。将 [`Program`][fatqat.Program] 传给 `run()`，再对立即完成的 [`Job`][fatqat.Job] 调用 `job.result()` 取得 [`Result`][fatqat.Result]。它们是解析脉冲的物理仿真器，而不是 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 的运行模式。

若要在物理 `|0>, |1>, |r>` 模型中使用校准门或选定位点的直接控制，请使用 [`Atom3LevelEmulator`][fatqat.emulator.Atom3LevelEmulator]。若要在 `|g>, |r>` 模型中直接编写全局控制，请使用 [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator]。比较和可执行工作流程参阅[选择并运行中性原子工作流程](../guide/neutral-atom-emulation.md)。

## 排列与程序资源


两个后端都要求规则 [`AtomArrangement`][fatqat.emulator.AtomArrangement]。坐标按行优先排列，为 `(column * spacing, row * spacing, 0)`；当前原子模型将间距解释为微米。程序必须为每个位点恰好声明一个二维量子资源；声明顺序将资源绑定到坐标。排列描述固定几何结构，不跟踪原子装载或损失。
`arrangement.num_sites` 和 `len(arrangement)` 都返回坐标数量，脉冲程序必须与其精确匹配。
相比之下，`AtomArraySimulator(num_sites=6)` 声明门级设备容量上限为六个资源，并接受资源数不超过六的程序；省略其 `num_sites` 参数则使该模拟器不设上限。

::: fatqat.emulator.AtomArrangement
    options:
      members:
        - "chain"
        - "rectangular"
        - "num_sites"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

## 运行配置与结果


两个 `run()` 方法的签名均为 `(program, *, shots=1024, resource_layout=None, simulation_config=None, result_config=None)`。可选布局仍必须恰好覆盖每个排列位点一次；默认使用声明顺序。验证错误会在返回作业前直接从 `run()` 抛出。执行开始后的故障由失败作业表示，`job.result()` 会引发 [`BackendExecutionError`][fatqat.errors.BackendExecutionError]。

`simulation_config` 只接受 `seed` 和 `schedule_mode`：

**模拟配置**

| 键 | 类型 | 默认值 | 效果与约束 |
| --- | --- | --- | --- |
| `seed` | `int` 或 `None`；不能是 `bool` | `None` | 用于测量、读出和轨迹采样的随机种子。整数必须非负；`None` 选择新种子。 |
| `schedule_mode` | `"ASAP"` 或 `"ALAP"` | `"ASAP"` | 在依赖关系允许的范围内尽早或尽晚放置操作。 |

`result_config` 只接受以下键：

**结果配置**

| 键 | 类型 | 默认值 | 效果与约束 |
| --- | --- | --- | --- |
| `counts` | `bool` 或 `None` | `None` | `True` 请求经典计数，`False` 禁用计数，`None` 在存在测量时启用。计数要求 `shots` 为正整数。 |
| `final_state` | `bool` 或 `None` | `None` | `True` 请求模型和模式专用的终态，`False` 禁用，`None` 在没有测量时启用。存在物理测量时，要求 `shots == 1`。 |

两个配置参数都必须是 `dict` 或 `None`；未知键会被拒绝。

每次运行都从固定直积态开始：三能级后端为 `|0>`，二能级后端为 `|g>`。两个构造函数都不接受 `initial_state` 参数。

可用的最终状态结果取决于后端和执行模式：

**最终状态表示**

| 后端／模式 | 结果访问器 | `N` 个位点的形状 | 解释 |
| --- | --- | --- | --- |
| 三能级，无测量 | [`get_density_matrix`][fatqat.Result.get_density_matrix] | `(3**N, 3**N)` | 精确系综状态。 |
| 三能级，有测量 | 请求时使用 [`get_density_matrix`][fatqat.Result.get_density_matrix] | `(3**N, 3**N)` | 一个采样后验状态；要求 `shots == 1`。 |
| 二能级，无 Lindblad 演化 | [`get_statevector`][fatqat.Result.get_statevector] | `(2**N,)` | 纯态；测量后则是一个采样后验状态。 |
| 二能级，未测量的 Lindblad 演化 | [`get_density_matrix`][fatqat.Result.get_density_matrix] | `(2**N, 2**N)` | 精确系综状态。 |
| 二能级，已测量的 Lindblad 演化 | 请求时使用 [`get_statevector`][fatqat.Result.get_statevector] | `(2**N,)` | 一条采样轨迹及后验状态；要求 `shots == 1`。 |

代码需要处理多种执行模式时，请使用 `result.available_data`。两个后端都不公开 QuTiP 值。

结果元数据标识模型格式、类别、ID 和修订版本。

## 三能级原子仿真器


[`Atom3LevelEmulator`][fatqat.emulator.Atom3LevelEmulator] 接受物理模型和排列。其默认门映射支持 [`RX`][fatqat.operations.RX]、[`RY`][fatqat.operations.RY]、[`RZ`][fatqat.operations.RZ] 和 [`CZ`][fatqat.operations.CZ]，也支持测量、重置、屏障和经典条件。

局部物理基为 `|0>, |1>, |r>`。在应用二元读出混淆前，测量会把这些能级映射到 `0, 1, 1`。完整量子三能级状态会保留；`|r>` 是相干泄漏，而不是物理原子损失。

带符号的 `C6/R^6` 漂移包含每一对已占据位点。校准 CZ 脉冲是固定的，不会在间距或 `C6` 变化时重新调谐。内置二元 `2 x 2` 经典读出混淆。默认 Lindblad 映射为空；传入映射可以添加兼容的量子三能级速率或时间噪声。量子三能级振幅阻尼需要两个相邻能级速率。速率使用每微秒，`t1`、`t2` 和 `t_phi` 使用微秒。接受背景噪声和普通操作域噪声。受支持形式参阅[脉冲仿真器](noise/backend-support.md#noise-emulator-support)，脉冲仿真器要求速率的原因参阅[连续时间噪声](pulse-control/index.md#pulse-probability-noise)。

### 构造与执行


::: fatqat.emulator.Atom3LevelEmulator
    options:
      members:
        - "model"
        - "arrangement"
        - "run"
        - "propagator"
        - "validate_noise_model"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

`propagator()` 返回相干完整量子三能级 `(3**N, 3**N)` 算符。它拒绝测量、重置和条件。`apply_final_frame=True` 包含最终虚拟参考系变换，`False` 省略该变换。仅读出噪声不会影响传播子。

### 模型与校准值


`Atom3LevelModel.from_document(...)` 解析与模型模式精确匹配的已解码映射。使用 `load_model_document("atom3level.reference")` 获取打包参考；自定义文件应先加载并解码，再传入映射。校准类接受自身的已解码校准映射。模型定义物种、基与跃迁、量单位、质量和带符号的 `C6`；校准提供 Raman 和 CZ 配方值。

### 类 `fatqat.emulator.Atom3LevelModel` { #fatqat.emulator.Atom3LevelModel }

使用 [`from_document`][fatqat.emulator.Atom3LevelModel.from_document] 创建实例；不支持直接构造。

::: fatqat.emulator.Atom3LevelModel.from_document

::: fatqat.emulator.Atom3LevelModel.control

::: fatqat.emulator.Atom3LevelModel.available_controls

::: fatqat.emulator.Atom3LevelModel.frame

::: fatqat.emulator.Atom3LevelModel.time_unit

::: fatqat.emulator.Atom3LevelCalibration
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.default_atom_3level_calibration

::: fatqat.emulator.default_atom_3level_gate_implementation_map

标准构建器要求 `model=` 和 `calibration=`，并返回新的 [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap]。其规则使用该模型的通道和参考系。几何结构和 C6 会影响物理演化，但不会重新调谐内置脉冲形状。

## 二能级原子仿真器


[`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] 要求一个 `Atom2LevelModel` 和一个排列。其全局驱动和失谐通道作用于每个位点；如何添加直接脉冲块参阅 [PulseOperation](pulse-control/pulse-operation.md)。脉冲程序之后可以跟一次终端测量，屏障会被忽略。内置门映射为空，因此普通门需要自定义映射。不支持重置、条件、逐位点控制、中途测量和测量后的脉冲。

### 构造与执行


::: fatqat.emulator.Atom2LevelEmulator
    options:
      members:
        - "model"
        - "arrangement"
        - "interaction_cutoff"
        - "run"
        - "propagator"
        - "validate_noise_model"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

`propagator()` 返回相干 `(2**N, 2**N)` 算符。它拒绝测量，并且在程序持续时间非零时拒绝 Lindblad 噪声。零持续时间程序即使有此类噪声也返回恒等算符，因为没有经过时间。

### 模型与控制


模型固定基顺序 `("g", "r")`、单位拼写、带符号的 `C6`、`C6/R^6` 相互作用定律和可选通道边界。它不包含几何结构或校准。

### 类 `fatqat.emulator.Atom2LevelModel` { #fatqat.emulator.Atom2LevelModel }

使用 [`from_document`][fatqat.emulator.Atom2LevelModel.from_document] 创建实例；不支持直接构造。

::: fatqat.emulator.Atom2LevelModel.from_document

::: fatqat.emulator.Atom2LevelModel.control

::: fatqat.emulator.Atom2LevelModel.available_controls

::: fatqat.emulator.Atom2LevelModel.angular_frequency_unit

::: fatqat.emulator.Atom2LevelModel.time_unit

全局驱动接受复数 [`SampledWaveform`][fatqat.emulator.SampledWaveform]；其复数值同时编码振幅和相位。全局失谐接受实数采样。二者都使用 `rad/us`，并作用于每个排列位点。

### 相互作用截断


默认的 `interaction_cutoff=None` 保留每一对，并保持完整的 `C6/R^6` Hamiltonian。有限非负截断会保留 Euclidean 距离不大于该值的位点对，距离使用模型距离单位（当前为微米）；`0.0` 禁用成对相互作用。对于矩形排列，`interaction_cutoff=arrangement.spacing` 只保留水平和垂直最近邻对。这是数值 Hamiltonian 截断，而不是物理阻塞半径。

### Lindblad 噪声与结果类型


内置形式列于[脉冲仿真器](noise/backend-support.md#noise-emulator-support)。每个背景注册指定一个位点；若要在多个位点应用同一种噪声，需要显式枚举位点。速率使用每微秒，弛豫时间使用微秒。有限 `p` 形式不会按脉冲持续时间转换。二元 [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] 是仅在物理坍缩后应用于报告数字的经典报告通道，不是 Lindblad 算符。

省略 `lindblad_implementation_map` 时，后端使用振幅阻尼、相位阻尼、热弛豫和退极化噪声的内置规则。这些默认规则只接受背景声明。显式映射会替换这些规则，并可为其注册的通道类型启用操作域速率声明。

没有 Lindblad 注册时，二能级后端使用纯态演化。未测量的含噪程序返回精确系综密度矩阵。带终端测量的含噪程序使用带种子的轨迹。零时间测量程序会在没有时间演化的情况下采样初态。即使是零速率 Lindblad 声明也会选择含噪结果类型。

直接编写脉冲参阅[脉冲控制](pulse-control/index.md)，完整二能级工作流程参阅[选择并运行中性原子工作流程](../guide/neutral-atom-emulation.md)。
