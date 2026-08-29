---
title: 操作
---
<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 操作


将 `fatqat.operations` 导入为 `ops`，然后用 [`fatqat.Program.add`][fatqat.Program.add] 向程序添加操作：

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2)
program.add(ops.H, 0)            # ready-to-use operation
program.add(ops.RX(0.2), 1)      # parameterized operation
program.add(ops.CX, (0, 1))      # ordered targets
```

`ops.H`、`ops.Reset` 等无参数操作无需括号即可直接使用。参数化门和 [`PulseOperation`][fatqat.operations.PulseOperation] 值必须先构造，再添加到程序中。使用 [`measure`][fatqat.Program.measure] 或 [`measure_all`][fatqat.Program.measure_all] 创建测量。

## 参考页面


**操作类别**

| 页面 | 内容 |
| --- | --- |
| [量子比特门](operations/qubit-gates.md) | 固定和参数化量子比特门、精确的目标顺序、矩阵以及构造函数参考。 |
| [量子多能级门](operations/qudit-gates.md) | 量子多能级门、能级约束和基态作用。 |
| [测量与结构操作](operations/structural.md) | 测量和重置行为，以及编译器屏障。 |
| [原子阵列操作](operations/atom-gates.md) | 原子阵列占据、配对及附加噪声约束。 |
| [PulseOperation](pulse-control/pulse-operation.md) | 按通道寻址的 `PulseOperation`——仍从 `fatqat.operations` 导入——及其时序和后端支持。 |

- [量子比特门](operations/qubit-gates.md)
- [量子多能级门](operations/qudit-gates.md)
- [测量与结构操作](operations/structural.md)
- [原子阵列操作](operations/atom-gates.md)

## 构造


对于基于目标的操作，[`add`][fatqat.Program.add] 会解析目标引用、检查元数并拒绝重复的标量目标。提交程序时，所选后端会检查操作和设备支持；不受支持的操作类别会引发 [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]。直接使用 [`PulseOperation`][fatqat.operations.PulseOperation] 时，应遵循 [PulseOperation](pulse-control/pulse-operation.md) 页面上的通道寻址规则，并且添加时不指定目标。

大多数目标是标量 [`RegisterRef`][fatqat.RegisterRef] 或整数。程序仅有一个量子寄存器时可以使用整数；有多个寄存器时，请索引所需寄存器。受控门采用控制位优先的顺序，各类别页面的矩阵和基态作用均将第一个局部操作数视为最高有效位。

[`RX`][fatqat.operations.RX]、[`RY`][fatqat.operations.RY] 和 [`RZ`][fatqat.operations.RZ] 接受一个 [`RegisterView`](registers.md#fatqat.RegisterView)；[`CX`][fatqat.operations.CX] 和 [`CZ`][fatqat.operations.CZ] 接受两个兼容的视图，并按顺序配对其中的成员。视图兼容规则参阅[寄存器](registers.md)，常规构造工作流程参阅[使用 Program 编写量子计算](../guide/program.md)。

## 操作基类


继承 [`Operation`][fatqat.operations.Operation] 可以定义新的程序级值，但不会注册矩阵或脉冲实现。自定义矩阵契约参阅[矩阵实现](implementation.md)，脉冲实现参阅[门实现](pulse-control/gate-realization.md)。

::: fatqat.operations.Operation
    options:
      members:
        - "name"
        - "num_subsystems"
        - "min_targets"
        - "accepts_views"
        - "validate_targets"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
