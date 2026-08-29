
# 噪声模型


[`NoiseModel`][fatqat.NoiseModel] 收集噪声规则，并说明每条规则的作用位置：匹配操作上、整个脉冲时间内，或报告测量结果时。一个模型可以组合模拟器通道、仿真器 Lindblad 算符、载体损失和读出混淆。每个后端只接受它能够实现的规则。

## 构建模型


下面的示例在每个 `RX` 后添加相位阻尼，在每个 `CZ` 的第二个操作数上添加振幅阻尼，并在 `q[0]` 上添加读出混淆。逻辑目标使用程序中的引用；设备标签目标使用运行时 [`ResourceLayout`][fatqat.ResourceLayout] 中的标签。

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
q = program.quantum_registers[0]
program.add(ops.RX(0.4), q[0])
program.add(ops.CZ, (q[0], q[1]))
program.measure_all()

noise = fq.NoiseModel()
noise.add(
    fq.noise.PhaseDamping(p=0.01),
    operation=ops.RX,
)
noise.add(
    fq.noise.AmplitudeDamping(p=0.002),
    operation=ops.CZ,
    target_positions=1,
)
noise.add(
    fq.noise.ReadoutConfusion(
        [[0.98, 0.04], [0.02, 0.96]]
    ),
    targets=q[0],
)

simulator = fq.simulator.Simulator(method="DM", noise=noise)
result = simulator.run(
    program,
    shots=1_000,
    simulation_config={"seed": 7},
    result_config={"counts": True},
).result()
```

## 噪声的作用位置


对于绑定到匹配操作的噪声，传入 `operation`。模拟器会在操作后应用通道或采样损失；仿真器会在脉冲期间保持匹配 Lindblad 算符有效。省略 `operation` 表示背景噪声，并不表示“作用于每个门”。读出混淆总是在测量时应用。

**噪声作用域**

| 作用域 | 噪声类型 | `operation` | `targets` | `target_positions` |
| --- | --- | --- | --- | --- |
| 匹配操作上 | [`Channel`][fatqat.noise.Channel] 或 [`Loss`][fatqat.noise.Loss] | 必需的操作实例或子类 | 可选的精确有序目标选择器 | 操作目标顺序中可选的受影响位置 |
| 脉冲时间内（背景） | 作用于一个子系统的 [`Channel`][fatqat.noise.Channel] | 省略或 `None` | 恰好一个逻辑引用或设备标签 | 必须省略或为 `None` |
| 测量时（读出） | [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | 必须省略 | 省略表示通用读出，或指定一个标量选择器 | 必须省略 |

对于读出混淆，`operation` 和 `target_positions` 都必须不存在。显式传入其中任一关键字，即使值为 `None`，也会报错。

背景噪声在已用仿真器时间内使用局部 Lindblad 算符。[`Loss`][fatqat.noise.Loss]、概率形式的 [`Depolarizing`][fatqat.noise.Depolarizing] 以及其他不恰好作用于一个子系统的噪声类型不能如此使用。无论单个操作的经典条件如何，背景噪声始终有效。

`Barrier`、`Reset` 和直接 [`PulseOperation`][fatqat.operations.PulseOperation] 控制都没有可附加噪声的边界。`Put` 只接受 [`Loss`][fatqat.noise.Loss]；损失在装载后应用，用于模拟装载效率不足。各内置后端实现的作用域和噪声形式参阅[后端支持](backend-support.md#noise-backend-support)。

## 匹配操作


可以传入操作值或操作类。例如，`operation=ops.X` 匹配导出的 `X` 单例，`operation=ops.RX` 匹配任意 `RX(...)` 角度。参数不会缩小匹配范围，注册操作基类也不会包括其子类。

附加到操作的噪声遵循该操作的经典条件。若条件为假，操作及其噪声都不会运行。模拟器按注册顺序在操作后应用兼容通道；仿真器在匹配脉冲期间让兼容 Lindblad 算符共同保持有效。

## 匹配目标


`targets` 可以使用程序中的逻辑 [`RegisterRef`][fatqat.RegisterRef] 值，或资源布局中的可哈希设备标签。

**接受的 `targets` 形式**

| 形式 | 作用域 | 含义与约束 |
| --- | --- | --- |
| `None` | 操作 | 匹配精确类的每个操作。 |
| 一个 `RegisterRef` 或非元组可哈希标签 | 操作 | 单元素选择器的简写。宽度已知的操作必须接受一个目标；可变元操作稍后检查。 |
| 非空 `RegisterRef` 值元组 | 操作 | 精确的有序逻辑目标选择器。长度必须等于已知操作宽度。 |
| 非空可哈希设备标签元组 | 操作 | 精确的有序设备标签选择器。长度必须等于已知操作宽度。 |
| 一个标量或单元素元组 | 背景 | 恰好选择一个逻辑引用或设备标签。 |
| `None` | 读出 | 选择每个被测操作数。通用和定向读出注册不能共存。 |
| 一个 `RegisterRef` 或非元组可哈希标签 | 读出 | 选择一个被测逻辑引用或设备标签。不支持相关读出。 |

不接受列表和 [`RegisterView`](../registers.md#fatqat.RegisterView) 值。目标元组不能混合逻辑引用和设备标签。FATQAT 始终把元组视为完整有序选择器，因此元组值设备标签需要再嵌套一层：`targets=(("site", 0),)` 选择一个这样的标签，`targets=(("site", 0), ("site", 1))` 选择一个有序对。读出噪声不能以元组值设备标签为目标。

选择器顺序必须与程序中的操作目标顺序一致。针对 `(q[0], q[1])` 的噪声规则不会匹配作用于 `(q[1], q[0])` 的同一操作。逻辑选择器和设备标签选择器使用相同的顺序约定。

## 选择受影响的操作数


操作匹配后，`target_positions` 决定哪些操作数接收噪声。传入一个整数或由非负整数构成的非空严格递增元组。位置使用从零开始的操作顺序；`None` 选择每个操作数。

```python
local_noise = fq.NoiseModel()
local_noise.add(
    fq.noise.AmplitudeDamping(p=0.002),
    operation=ops.CZ,
    target_positions=0,
)
local_noise.add(
    fq.noise.AmplitudeDamping(p=0.003),
    operation=ops.CZ,
    target_positions=1,
)
```

选中操作数的数量必须与噪声类型作用的子系统数量相同。若两个大小都已知，[`add`][fatqat.NoiseModel.add] 会立即检查；可变元操作在程序运行时检查。

## 组合噪声源


[`add`][fatqat.NoiseModel.add] 只会追加，绝不会替换已有注册。不同噪声类型可以作用于同一操作。同一类型也可以用于互不相交的目标或操作数位置。背景噪声与操作噪声彼此独立，因此同一类型可以同时出现在两个作用域中。

对于同一操作或背景作用域，FATQAT 会拒绝同类型的重叠规则。匹配所有目标的规则与精确目标选择器重叠，选择每个操作数也会与位置选择重叠。

在资源布局可用之前，无法比较逻辑选择器和设备标签选择器。因此可以同时添加二者，但如果布局使它们为实际操作选择了相同的噪声类型和操作数，执行会引发 [`BackendValidationError`][fatqat.errors.BackendValidationError]。

每个被测操作数的读出规则必须唯一。重复的通用或精确选择器会在 `add` 期间失败；通用注册和定向注册不能混合。若逻辑选择器和设备标签选择器指向同一被测操作数，程序运行时会拒绝。

## 验证时机


**验证阶段**

| 时机 | 检查内容 | 典型故障 |
| --- | --- | --- |
| 创建噪声值 | 概率、速率、矩阵、有限值及类型专用参数关系 | 噪声类型抛出的 `TypeError` 或 `ValueError` |
| [`add`][fatqat.NoiseModel.add] | 噪声类型、操作边界、目标形式、已知宽度、位置顺序／范围和立即可见的冲突 | `TypeError` 或 `ValueError`；失败的添加不会改变模型 |
| 创建后端 | 已配置后端是否接受每一种噪声形式和作用域 | [`BackendValidationError`][fatqat.errors.BackendValidationError] |
| 运行程序 | 引用是否属于程序、设备标签是否属于布局、选择器能否无冲突解析、维度和所选执行方法是否兼容 | 通常为 [`BackendValidationError`][fatqat.errors.BackendValidationError] 或 [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] |

有效但没有匹配操作或测量的选择器不产生作用，也不会报错。某些检查需要具体程序、资源布局和执行方法。

## 验证后端支持


对于模拟器，使用 [`validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model]；对于脉冲仿真器，使用 [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] 来验证候选模型。已配置后端接受该模型时，方法返回 `None`；否则会引发 [`BackendValidationError`][fatqat.errors.BackendValidationError]，消息中包含所有模型级拒绝原因。

```python
probe = fq.simulator.Simulator(method="DM")
probe.validate_noise_model(noise)
```

依赖程序和布局的检查仍会在使用模型时执行。

## API


::: fatqat.NoiseModel
    options:
      members:
        - "add"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
