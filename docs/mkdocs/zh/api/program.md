<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# Program


[`Program`][fatqat.Program] 记录量子工作负载，而不将其绑定到具体设备。按执行顺序添加操作和测量，然后在运行时选择后端。

构建程序时，FATQAT 会捕获格式错误的目标、测量和条件。后端会检查其是否支持请求的操作、维度、放置和前馈。

```python
import fatqat as fq
import fatqat.operations as ops

bell = fq.Program(2, 2, metadata={"name": "bell"})
bell.add(ops.H, 0)
bell.add(ops.CX, (0, 1))
bell.measure_all()
```

## 寄存器


两个寄存器参数接受以下形式：

**寄存器输入**

| 形式 | 结果 | 约束 |
| --- | --- | --- |
| 非负整数 `n` | `n > 0` 时创建一个名为 `"q"` 或 `"c"` 的二维寄存器；`0` 时不创建寄存器。 | 值必须是严格意义上的整数而非布尔值；负值会被拒绝。 |
| 寄存器列表或元组 | 将给定寄存器对象原样保存在元组中。 | 每一项都必须是相应类别的量子或经典寄存器。需要名称、多个寄存器、网格或量子多能级系统时使用此形式。 |

显式寄存器、维度和网格选择参阅[寄存器](registers.md)。

## 目标


裸整数索引相应类别中唯一的寄存器，而不是跨多个寄存器的全局索引。存在多个寄存器时，请索引所需寄存器，并传入得到的 [`RegisterRef`][fatqat.RegisterRef]。

**目标形式**

| 形式 | 接受位置 | 规则 |
| --- | --- | --- |
| 整数 | [`add`][fatqat.Program.add]、[`measure`][fatqat.Program.measure] 和条件 | 相应类别必须恰好有一个寄存器，且值必须处于从零开始的边界内。 |
| [`RegisterRef`][fatqat.RegisterRef] | [`add`][fatqat.Program.add]、[`measure`][fatqat.Program.measure] 和条件 | 必须是所需的寄存器类别，且来自此程序中的寄存器。 |
| [`RegisterView`](registers.md#fatqat.RegisterView) | 仅 [`add`][fatqat.Program.add] | [`RX`][fatqat.operations.RX]、[`RY`][fatqat.operations.RY] 和 [`RZ`][fatqat.operations.RZ] 接受一个视图。[`CX`][fatqat.operations.CX] 和 [`CZ`][fatqat.operations.CZ] 接受两个兼容视图。测量不接受视图。 |

[`PulseOperation`][fatqat.operations.PulseOperation] 不使用上述目标形式。请以 `program.add(operation)` 添加，不传 `targets` 参数。详见 [PulseOperation](pulse-control/pulse-operation.md)。

对于其他操作，`targets` 是一个按操作数顺序排列的元组。受控门把控制位列在目标位之前。视图选择和配对参阅[寄存器](registers.md)，常规构造工作流程参阅[使用 Program 编写量子计算](../guide/program.md)。

## 条件


向 [`add`][fatqat.Program.add] 传入 `condition=(slot, literal)`，或者传入由这些对组成的非空元组或列表以表示逻辑与。槽位遵循与其他经典操作数相同的整数或引用规则。每个字面量都是满足 `0 <= literal < slot.dim` 的 Python 整数；也接受布尔值。执行到该操作时，会将每一项与当前经典值比较。

添加操作时，FATQAT 会检查每个条件。条件可以引用尚未测量、初始值为零的槽位；此前的测量会替换该值。是否支持前馈由后端决定。

## 测量


[`measure`][fatqat.Program.measure] 按位置配对量子目标与经典输出。两侧都必须非空、长度相同，并且每个位置的维度一致。重复操作数按配对顺序处理；测量行为参阅[测量与结构操作](operations/structural.md)。

[`measure_all`][fatqat.Program.measure_all] 按声明顺序展平所有寄存器及其成员，并追加一次分组测量。它要求量子和经典计数相等且非零，并且每个位置的维度一致。包含程序中途测量与前馈的工作流程参阅[使用 Program 编写量子计算](../guide/program.md)。

<a id="program-templates"></a>


## 参数绑定


参数是不可变的身份对象。名称只是标签：两个 `Parameter("theta")` 对象是不同的绑定键。若多个操作参数应共享同一值，请复用同一个对象。

**绑定形式**

| 映射键 | 接受的值 | 约束 |
| --- | --- | --- |
| [`Parameter`][fatqat.Parameter] | 内置整数或浮点数，或 NumPy 整数、浮点标量 | 必须把同一对象直接提供给某个操作参数。 |
| [`ParameterVector`][fatqat.ParameterVector] | 一维 NumPy 数组，或由接受标量组成的非字符串、非字节串、非映射可迭代对象 | 按迭代顺序消费一次。值的长度必须匹配，并且向量中的每个元素都必须直接用作操作参数。若要部分绑定向量，请绑定单个元素。 |

[`assign_parameters`][fatqat.Program.assign_parameters] 接受空映射或部分映射，并返回新程序。它只绑定直接用作操作参数的 [`Parameter`][fatqat.Parameter] 对象。未绑定参数保持符号形式，数值执行和导出时会被拒绝。不接受字符串键、位置赋值、布尔值或复数值，也不能同时为向量及其中某个元素赋值。替换值仍需通过操作的常规验证。编写工作流程参阅[使用 Program 编写量子计算](../guide/program.md)，参数扫描参阅[模拟量子程序](../guide/simulation.md)。完整的绑定与执行契约在此处及 [Simulator](simulator.md) 中规定。

[`copy`][fatqat.Program.copy] 和 [`assign_parameters`][fatqat.Program.assign_parameters] 返回新程序。[`add`][fatqat.Program.add]、[`measure`][fatqat.Program.measure] 和 [`measure_all`][fatqat.Program.measure_all] 更新当前程序并返回 `None`。

## 绘制


FatQat 的电路绘制基于 QuTiP-QIP 的电路绘制工具。`Program.draw()` 会先把 Program 指令转换为渲染适配器，再调用所选 QuTiP-QIP 渲染器。

**渲染器**

| `renderer` | 返回值 | 说明 |
| --- | --- | --- |
| `"matplotlib"`（默认） | Matplotlib `Figure` | 传入 `ax=` 可在现有坐标轴上绘制；其他关键字参数会转发给渲染器。 |
| `"text"` | 终端图示字符串 | 返回该字符串，而不是打印。 |
| 其他 QuTiP-QIP 渲染器名称 | 由渲染器定义 | 名称和关键字参数原样转发。 |

电路图为每个槽位使用一条线路，但不显示寄存器维度。未知或自定义操作显示为带标签的方框。直接 [`PulseOperation`][fatqat.operations.PulseOperation] 无法表示，会引发 [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]。

只有在与 QuTiP-QIP 绘制工具做底层集成时才使用 [`fatqat.draw.to_qubit_circuit`][fatqat.draw.to_qubit_circuit]。返回的电路是渲染适配器，而非执行对象。

::: fatqat.draw.to_qubit_circuit

## 参考


::: fatqat.Program
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:add|measure|measure_all|draw|copy|assign_parameters)$"

::: fatqat.Program.add

::: fatqat.Program.measure

::: fatqat.Program.measure_all

::: fatqat.Program.draw

::: fatqat.Program.copy

::: fatqat.Program.assign_parameters

## 参数值


::: fatqat.Parameter
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.ParameterVector
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
