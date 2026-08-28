<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 寄存器


寄存器为量子和经典程序槽位提供稳定身份。使用 `Program(quantum_count, classical_count)` 时，每个正计数都会创建一个名为 `"q"` 或 `"c"` 的默认寄存器；零则不创建该类别的寄存器。需要其他名称、多个寄存器、网格、元数据或大于二的局部维度时，请显式构造寄存器。

## 寄存器类型


**寄存器选择**

| 类型 | 程序中的作用 | 大小 |
| --- | --- | --- |
| [`Register`][fatqat.Register] | 公共基类；本身不能作为 [`Program`][fatqat.Program] 的量子或经典寄存器 | 显式指定正 `size` |
| [`QuantumRegister`][fatqat.QuantumRegister] | 量子操作和测量的目标 | 显式指定正 `size` |
| [`ClassicalRegister`][fatqat.ClassicalRegister] | 测量输出和条件值 | 显式指定正 `size` |
| [`GridRegister`][fatqat.GridRegister] | 带矩形选择辅助方法的量子目标 | 由 `rows * cols` 推导 |

名称只是标签，无需唯一。请保留传给 [`Program`][fatqat.Program] 的同一寄存器对象，并从该对象索引；字段相同的新建寄存器不能互换。`metadata` 是用于应用数据的可变字符串键映射。

用 `register[index]` 索引会创建不可变的 [`RegisterRef`][fatqat.RegisterRef]。程序包含多个寄存器时，请传入显式引用，而不是有歧义的整数：

```python
import fatqat as fq
import fatqat.operations as ops

left = fq.QuantumRegister(2, name="left")
right = fq.QuantumRegister(2, name="right")
program = fq.Program([left, right])
program.add(ops.H, right[0])
```

`dim=2` 创建量子比特或经典比特。更大的值会创建量子多能级系统或 d 进制经典数字；每一对测量对象的量子维度和经典维度必须相同。构造寄存器时接受所有不小于二的整数维度，但单个操作和后端可能只支持其中一部分。混合量子比特—量子三能级示例参阅[使用 Program 编写量子计算](../guide/program.md)。

::: fatqat.Register
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "^(?:__getitem__)$"

::: fatqat.QuantumRegister
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.ClassicalRegister
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.RegisterRef
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

## 网格选择


[`GridRegister`][fatqat.GridRegister] 按行优先顺序排列逻辑目标，但不为其分配物理坐标。`(row, col)` 的展平索引是 `row * cols + col`，其选择辅助方法返回 [`RegisterView`](#fatqat.RegisterView) 对象，而不是引用元组。

对于 `GridRegister(2, 3)`，辅助方法选择以下展平索引：

**网格选择顺序**

| 表达式 | 按顺序选中的索引 |
| --- | --- |
| `grid.all()` | `(0, 1, 2, 3, 4, 5)` |
| `grid.row(1)` | `(3, 4, 5)` |
| `grid.column(1)` | `(1, 4)` |
| `grid.block((0, 2), (1, 3))` | `(1, 2, 4, 5)` |

将视图传给 [`add`][fatqat.Program.add]。内置的视图兼容操作包括 [`RX`][fatqat.operations.RX]、[`RY`][fatqat.operations.RY]、[`RZ`][fatqat.operations.RZ]、[`CX`][fatqat.operations.CX] 和 [`CZ`][fatqat.operations.CZ]。一元操作独立作用于每个选中成员；[`CX`][fatqat.operations.CX] 和 [`CZ`][fatqat.operations.CZ] 按顺序配对两个视图的对应成员。一对视图必须采用相同类别的网格选择且基数相同，同一网格上的选择不能重叠。测量和 QASM 导出要求标量目标。后端负责验证物理放置和连接性。常规 Program 工作流程参阅[使用 Program 编写量子计算](../guide/program.md)，物理放置参阅[针对硬件配置测试 Program](../guide/hardware-profile-simulation.md)。

::: fatqat.GridRegister
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

<a id="fatqat.registers.RegisterView"></a>
### 类 `fatqat.RegisterView` { #fatqat.RegisterView }

由网格选择辅助方法返回的不可变、可哈希目标。它的 [`register`](#fatqat.RegisterView.register) 属性标识所选网格。请通过网格辅助方法获取视图；不支持直接构造。

<a id="fatqat.registers.RegisterView.register"></a>
#### 属性 `register` { #fatqat.RegisterView.register }

**类型：** `fatqat.GridRegister`

包含所选成员的网格寄存器。


## 资源布局


[`ResourceLayout`][fatqat.ResourceLayout] 将标量量子 [`RegisterRef`][fatqat.RegisterRef] 操作数与设备标签关联。大多数应用可以使用后端默认布局；需要指定受支持的放置时，传入 `resource_layout=`。各后端会定义其接受的标签，并在程序运行时检查覆盖范围、唯一性、维度、放置和连接性。[`PulseOperation`][fatqat.operations.PulseOperation] 的通道直接寻址仿真器模型，不使用此布局。

::: fatqat.ResourceLayout
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

### 类型 `fatqat.DeviceOperand` { #fatqat.DeviceOperand }

由后端定义、表示单个设备资源的不透明可哈希标签。
