# 使用 Program 编写量子计算

[`Program`][fatqat.Program] 是你构建并传给每一个 FatQat 后端的对象。如果你
习惯使用 `Circuit` 类，这就是你要找的对象。FatQat 采用更宽泛的名称，是因为
Program 不只能够描述线路门：它还可以描述测量与经典条件、符号参数、量子比特
与量子多能级系统，以及直接物理控制。它按执行顺序记录逻辑系统及其指令，但不
预先选择后端如何执行这些指令。

先从熟悉的量子比特线路开始，再加入寄存器、前馈、参数、混合局域维数、绘图和
直接控制，而无需改变编程方式。

## 声明逻辑系统

对于小型量子比特线路，使用数量作为简写很方便：

```python
import fatqat as fq

program = fq.Program(2, 2)  # two qubits and two classical bits
```

显式寄存器会为资源赋予有意义的名称；当 Program 包含多个寄存器时，也必须
使用这种方式：

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> data = fq.QuantumRegister(2, name="data")
>>> readout = fq.ClassicalRegister(2, name="readout")
>>> program = fq.Program([data], [readout])
```

对寄存器建立索引会返回其中某个槽位的引用。随着 Program 变大，`data[0]`
这样的显式引用始终明确；只有在相应类型恰好只有一个寄存器时，裸整数才是方便
的写法。

!!! tip "需要按行、列操作？"

    [`GridRegister`][fatqat.GridRegister] 提供逻辑上的行选择和列选择：

    ```pycon
    >>> grid = fq.GridRegister(2, 3, name="grid")
    >>> grid_program = fq.Program([grid])
    >>> grid_program.add(ops.RX(0.2), grid.row(1))
    ```

    此处的一行是该操作的一组目标。它描述的是逻辑结构，而不是物理硬件上的
    布局；布局会在执行 Program 时选定。

## 按顺序组合操作

按照操作应发生的顺序添加它们：

```pycon
>>> program.add(ops.H, data[0])
>>> program.add(ops.RY(0.3), data[1])
>>> program.add(ops.CX, (data[0], data[1]))
```

`H` 和 `CX` 这样的固定门是可以直接使用的值。`RY` 这样的参数化门则用其
数值参数创建。多目标操作接收一个按操作数顺序排列的元组；对于 `CX`，控制位
在前，目标位在后。

这几个操作展示了调用模式。[操作 API 参考](../api/operations.md)列出了完整的
操作集合及其准确定义。

## 在 Program 内部观测并响应

测量同样是一条有序指令。它可以结束线路，也可以为后续条件提供经典值：

```pycon
>>> dynamic = fq.Program(2, 2)
>>> dynamic.add(ops.H, 0)
>>> dynamic.measure(0, 0)
>>> dynamic.add(ops.X, 1, condition=(0, 1))
>>> dynamic.add(ops.Reset, 0)
>>> dynamic.measure(1, 1)
```

在每次采样中，`X` 只有在前面的测量向经典比特 0 写入 `1` 时才作用于
量子比特 1。随后，重置会把量子比特 0 制备为 `|0>`，但不会擦除已存储的
经典值。最后一次测量表明，经典比特 1 会跟随比特 0：

```pycon
>>> counts = (
...     fq.simulator.Simulator()
...     .run(dynamic, shots=100, simulation_config={"seed": 7})
...     .result()
...     .get_counts()
... )
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
100
```

后端会决定自己能否保留这种程序中途行为。不支持测量、重置或经典前馈的后端
会拒绝该 Program，而不会悄悄改变其含义。

## 制作可复用模板

[`Parameter`][fatqat.Parameter] 可以代替操作的数值参数。绑定会返回一个新的
Program，原模板仍可用于其他值：

```pycon
>>> import numpy as np
>>> theta = fq.Parameter("theta")
>>> template = fq.Program(1)
>>> template.add(ops.RY(theta), 0)
>>> quarter_turn = template.assign_parameters({theta: np.pi / 2})
>>> half_turn = template.assign_parameters({theta: np.pi})
>>> template_backend = fq.simulator.Simulator("SV", runtime="numpy")
>>> quarter_state = template_backend.run(quarter_turn).result().get_statevector()
>>> half_state = template_backend.run(half_turn).result().get_statevector()
>>> round(float(abs(quarter_state[1]) ** 2), 3)
0.5
>>> round(float(abs(half_state[1]) ** 2), 3)
1.0
```

绑定使用参数对象，而不是它的显示名称。如果多个门应共享同一个值，请在其中
复用同一对象；若要创建显式排序的一组参数，请使用
[`ParameterVector`][fatqat.ParameterVector]。[模拟章节](simulation.md)会在
不重建模板的情况下执行一整批参数值。

## 混合量子比特与量子三能级系统

寄存器的 `dim` 是局域基态的数量。默认值 `2` 创建量子比特或经典比特；
`dim=3` 创建量子三能级系统或三值经典数字。不同维数可以共存于同一个混合
Program：

```pycon
>>> qubit = fq.QuantumRegister(1, name="qubit")
>>> qutrit = fq.QuantumRegister(1, name="qutrit", dim=3)
>>> bit = fq.ClassicalRegister(1, name="bit")
>>> trit = fq.ClassicalRegister(1, name="trit", dim=3)
>>> hybrid = fq.Program([qubit, qutrit], [bit, trit])
>>> hybrid.add(ops.X, qubit[0])
>>> hybrid.add(ops.Shift(2), qutrit[0])
>>> hybrid.measure(
...     (qubit[0], qutrit[0]),
...     (bit[0], trit[0]),
... )
```

`X` 将量子比特制备为 `|1>`，与维数无关的 `Shift(2)` 则把量子三能级系统
从 `|0>` 映射到 `|2>`。测量会把每个量子槽位与相同维数的经典槽位配对：

```pycon
>>> hybrid_result = (
...     fq.simulator.Simulator("SV")
...     .run(hybrid, shots=20, simulation_config={"seed": 7})
...     .result()
... )
>>> hybrid_result.get_counts_as_tuples()
{(1, 2): 20}
```

元组键按声明顺序列出扁平化后的经典槽位，明确表示量子比特值为 `1`、量子
三能级值为 `2`。每种操作自行决定接受哪些局域维数；超越 `Shift` 后，请
查阅[量子多能级操作参考](../api/operations/qudit-gates.md)。

## 检查已写入的内容

需要目视检查时，随时调用 `draw()`：

```python
diagram = dynamic.draw("text")
print(diagram)
```

Matplotlib 渲染器是默认选项；文本渲染器则返回字符串。二者展示的都是指令
结构，而不是执行过程。在本例中，条件操作被标记为 `X if c0=1`，因此运行
之前就能看到前馈。[快速上手](quickstart.md)展示了 Matplotlib 输出。

!!! important "重要"

    线路图为每个量子或经典槽位使用一条线，但不会显示寄存器的局域维数。
    因此，量子比特线和量子三能级线看起来相同。量子多能级操作和自定义操作
    显示为带标签的方框。此线路渲染器无法表示
    [`PulseOperation`][fatqat.operations.PulseOperation]。

直接物理控制仍以 `PulseOperation` 的形式进入同一个 Program。添加时不需要
逻辑目标，因为其控制通道已经标识了物理资源。[哈密顿量仿真章节](hamiltonian-emulation.md)
会改用波形与时间线呈现这些控制。

## Program 会校验什么

编写期间，`Program` 会检查结构性问题：引用是否属于它、目标数量是否匹配，
以及被测量的量子维数与经典维数是否一致。所选后端随后回答另一个问题：它能否
实现这些操作、维数、控制和经典行为。

继续阅读[选择物理建模的细致程度](execution-models.md)，了解一套编程接口如何
通向 FatQat 的不同执行路径。需要准确的接受形式或校验行为时，请查阅
[Program API](../api/program.md)。
