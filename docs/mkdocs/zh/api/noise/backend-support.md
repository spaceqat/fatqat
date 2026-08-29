
<a id="noise-backend-support"></a>


# 后端支持


下列表格说明各内置后端接受哪些噪声类型和形式。**内置**表示标准后端直接接受；**仅自定义**表示必须为该精确噪声类型提供实现映射；**不支持**表示自定义映射也无法在该后端类别上启用。

若要检查一个已配置的后端，请对模拟器使用 [`validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model]，对脉冲仿真器使用 [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model]。这些方法无需程序即可验证模型。运行具体程序时，FATQAT 还会检查引用、设备标签、操作匹配、维度和执行方法限制。

<a id="noise-simulator-support"></a>


## 模拟器


[`Simulator`][fatqat.simulator.Simulator]、两个超导量子比特配置和 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 共享以下通道规则。各配置可能施加额外的操作、放置和维度限制。

**内置模拟器通道**

| 噪声类型和形式 | 支持情况 |
| --- | --- |
| [`Depolarizing`][fatqat.noise.Depolarizing] `(p)` | **内置。**在匹配操作之后联合作用于选定操作数。 |
| [`PauliChannel`][fatqat.noise.PauliChannel] | **内置。**字符串宽度必须等于选中的量子比特操作数数量。 |
| [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(p)` | **内置。**作用于一个选定操作数；维度 $d$ 需要 $d-1$ 个相邻跃迁概率。 |
| [`PhaseDamping`][fatqat.noise.PhaseDamping] `(p)` | **内置。**作用于任意有限局部维度的一个选定操作数。 |
| 内置速率形式和 [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] | **不支持。**模拟器没有物理时间线，不会把速率或时间转换为一次通道应用。 |
| [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | **内置。**可通用于所有测量，也可作用于一个被测操作数。矩阵大小必须等于报告数字维度，因此超导量子比特配置要求 `2 x 2`。 |

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 还支持匹配操作之后的 [`Loss`][fatqat.noise.Loss]。它会独立采样每个选中且存在的载体。匹配注册即使 `p=0` 也会启用显式占据状态；`Put` 之后的损失可模拟装载失败。其他模拟器拒绝 `Loss`。空位点擦除结果 `2` 会绕过读出混淆，因为没有测得物理数字。

上述所有受支持的概率形式通道都必须附加到操作；矩阵后端拒绝背景通道。自定义 [`ChannelImplementationMap`][fatqat.noise.ChannelImplementationMap] 可以添加一种有限通道类型，或替换已有类型的规则。它无法绕过对内置速率形式、背景通道或 `ThermalRelaxation` 的拒绝。自定义类型的参数名称由其规则解释，规则结果仍表示一次有限通道应用。提供的映射会替换默认映射，并在构造后端时复制。

执行方法也会影响支持情况。`statevector` 为每个含噪 shot 采样一条 Kraus 轨迹，`density_matrix` 和 `superop` 精确应用概率形式通道。`unitary` 会拒绝任何与程序匹配的通道。只有 `statevector` 和 `density_matrix` 支持 AtomArray 占据状态生命周期；存在损失时，请求最终状态要求 `shots == 1`。

<a id="noise-emulator-support"></a>


## 脉冲仿真器


脉冲仿真器使用由受支持速率和时间构造的局部 Lindblad 算符，不会从概率推导速率。内置概率形式、[`PauliChannel`][fatqat.noise.PauliChannel] 和 [`Loss`][fatqat.noise.Loss] 即使在自定义 Lindblad 映射中注册了其类型也仍不受支持。

**仿真器支持**

| 后端 | 内置 Lindblad 算符行为 | 自定义映射与读出 |
| --- | --- | --- |
| [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] | 背景或操作作用域：三能级模型使用带两个速率的 [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(rate)`、[`PhaseDamping`][fatqat.noise.PhaseDamping] `(rate or t_phi)` 和 [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation]。 | [`Depolarizing`][fatqat.noise.Depolarizing] `(rate)` 在背景或操作作用域中**仅支持自定义**。读出混淆为内置且是二元的。虽然只调用 [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] 可能不会拒绝更大的方阵，但运行要求 `2 x 2` 矩阵。 |
| [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] | 背景作用域：[`Depolarizing`][fatqat.noise.Depolarizing] `(rate)`、带一个速率的 [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(rate)`、[`PhaseDamping`][fatqat.noise.PhaseDamping] `(rate or t_phi)` 和 [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation]。 | 显式映射还可在匹配操作窗口期间启用兼容噪声。读出混淆为内置且仅支持 `2 x 2`。 |
| [`Atom3LevelEmulator`][fatqat.emulator.Atom3LevelEmulator] | 无。默认 Lindblad 实现映射为空。 | 兼容的速率／时间噪声在背景或操作作用域中**仅支持自定义**；振幅阻尼需要两个速率。读出混淆为内置且仅支持 `2 x 2`，即使物理模型有三个能级也是如此。 |

每个背景规则选择一个位点。操作域噪声仅在匹配脉冲窗口期间生效。读出混淆可以作用于所有测量或一个操作数；不支持多操作数相关读出。

速率使用模型时间单位的倒数。`t_phi`、`t1` 和 `t2` 直接使用该单位。参考 [`time_unit`][fatqat.emulator.TransmonModel.time_unit] 为纳秒；[`time_unit`][fatqat.emulator.Atom2LevelModel.time_unit] 和 [`time_unit`][fatqat.emulator.Atom3LevelModel.time_unit] 为微秒。请读取所选模型的 `time_unit`，不要根据数值大小猜测。

显式 [`LindbladImplementationMap`][fatqat.noise.LindbladImplementationMap] 会替换仿真器的默认映射，仿真器会在构造时复制它。Atom2 的内置映射只支持背景噪声；自定义映射还可在匹配操作窗口期间启用兼容噪声。Atom3 的默认映射为空，因此所有动力学噪声都需要自定义规则。自定义映射仍无法启用内置概率形式、[`PauliChannel`][fatqat.noise.PauliChannel]、非局部噪声或 [`Loss`][fatqat.noise.Loss]。自定义通道类型必须描述连续生成元，其解释由已注册规则定义。参阅[自定义噪声实现](custom-implementations.md)。
