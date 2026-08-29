---
title: "制备并测量贝尔态"
description: "从精确振幅出发，得到固定随机种子下的测量计数，并将其与理想分布比较，完整追踪一个两比特贝尔态。"
---
<!-- 中文译文人工维护；运行结果由 docs/mkdocs/tools/convert_tutorials.py 从规范源码同步。 -->

# 制备并测量贝尔态

<div class="grid cards" markdown>

-   :material-map-marker-path: **学习路径**

    基础

-   :material-language-python: **可执行源码**

    [下载 `plot_bell_state.py`](../downloads/tutorials/plot_bell_state.py){ download }

</div>

本教程构建最小的纠缠量子系统，并从精确态矢量一路追踪到有限采样次数的测量数据。在这个过程中，我们会看到量子态与重复测量所收集的经典证据之间的区别。

对于两个量子比特，这里使用的贝尔态是

$$
|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}.
$$

这个态矢无法分解为两个独立的单比特态。单独测量任一量子比特时，得到零或一的概率相同，但两个结果完全关联。因此，理想采样应满足 $P(00)=P(11)=1/2$ 和 $P(01)=P(10)=0$。

我们先查看精确振幅，再加入测量并收集可复现的采样计数，最后在图中将观测频率与理论分布比较。

!!! info "基于源码的教程"

    说明文字是对版本库中教程源码的人工中文翻译，页面中的可执行单元保留规范源码。转换脚本从同一源码捕获运行结果；其中的英文标签来自源码的打印语句，保留原样以便核对。页面不显示仅用于文档验证的代码段。下载并直接运行 Python 文件即可复现图形与标准输出。

## 导入与显示设置

`fatqat.Program` 是与后端无关的电路描述。门的值位于 `fatqat.operations` 中；`H` 和 `CX` 这样的固定门可直接传入，无须实例化。NumPy 用于检查量子态，运行规范源码时，Matplotlib 会生成图形。

```python title="Python 单元 1"
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

np.set_printoptions(precision=3, suppress=True)
```

## 构建纠缠电路

系统从 $|00\rangle$ 开始。在第零个量子比特上施加 Hadamard 门后得到

$$
\frac{|00\rangle + |01\rangle}{\sqrt{2}},
$$

这里采用 fatqat 的小端子系统约定。随后，受控 X 门仅在第零个量子比特为一时翻转第一个量子比特，由此得到 $|\Phi^+\rangle$。第一个程序不含经典寄存器和测量，因为我们希望取得精确的最终态。

```python title="Python 单元 2"
bell_program = fq.Program(2)
bell_program.add(ops.H, 0)
bell_program.add(ops.CX, (0, 1))
```

## 检查精确态矢量

`method="SV"` 选择态矢量模拟。不含测量时，运行结果是确定的：执行一次后端即可得到精确的最终向量。我们禁用计数，并显式请求原生的最终态数据。

```python title="Python 单元 3"
backend = fq.simulator.Simulator(method="SV")
state_result = backend.run(
    bell_program,
    result_config={"counts": False, "final_state": True},
).result()
statevector = state_result.get_statevector()

print("Statevector:")
print(statevector)
print(f"Total probability: {np.vdot(statevector, statevector).real:.12f}")
```

<!-- tutorial-result-start:cell-3 -->
!!! example "运行结果"

    ```text
    Statevector:
    [0.707+0.j 0.   +0.j 0.   +0.j 0.707+0.j]
    Total probability: 1.000000000000
    ```

<!-- tutorial-result-end:cell-3 -->

基底顺序是 $|00\rangle$、$|01\rangle$、$|10\rangle$、$|11\rangle$。因此，打印向量在索引零和三处的振幅为 $1/\sqrt{2}$，其余处为零。规范源码还会用仅限验证的代码行检查这一预期；本页不在展示代码中列出这些检查。

```python title="Python 单元 4"
probabilities = np.abs(statevector) ** 2
print("Exact basis probabilities:", probabilities)
```

<!-- tutorial-result-start:cell-4 -->
!!! example "运行结果"

    ```text
    Exact basis probabilities: [0.5 0.  0.  0.5]
    ```

<!-- tutorial-result-end:cell-4 -->

## 添加经典测量

采样程序需要两个经典比特。我们重新构建这个短程序，保持其量子指令不变，然后将第零、第一个量子位分别测量到第零、第一个经典位。计数字符串将索引最高的经典位显示在左侧，因此关联结果显示为 `"00"` 和 `"11"`。

```python title="Python 单元 5"
measured_program = fq.Program(2, 2)
measured_program.add(ops.H, 0)
measured_program.add(ops.CX, (0, 1))
measured_program.measure((0, 1), (0, 1))
```

## 采样一次可复现的实验

有限采样的结果会在精确概率附近波动。固定随机种子可使教程的打印输出在多次运行中保持稳定；希望得到相互独立实验样本的应用可以省略它。随机种子控制采样过程，而不会改变量子门所制备的态。

```python title="Python 单元 6"
shots = 1_000
sample_result = backend.run(
    measured_program,
    shots=shots,
    simulation_config={"seed": 7},
).result()
counts = sample_result.get_counts()

print("Seeded counts:", counts)
print("Available result data:", sorted(sample_result.available_data))
```

<!-- tutorial-result-start:cell-6 -->
!!! example "运行结果"

    ```text
    Seeded counts: {'00': 502, '11': 498}
    Available result data: ['counts']
    ```

<!-- tutorial-result-end:cell-6 -->

## 将观测结果与理论比较

对于输出 $x$，经验频率

$$
\hat{P}(x) = \frac{N_x}{N}
$$

是对 Born 概率的估计。虚线标出两个允许输出的理想概率 $1/2$。采样噪声会使每个柱子稍微偏离这个值，而在无噪声电路中，不可能出现的 `01` 和 `10` 结果仍不会出现。

```python title="Python 单元 7"
outcomes = ("00", "01", "10", "11")
observed = np.array([counts.get(outcome, 0) / shots for outcome in outcomes])
ideal = np.array([0.5, 0.0, 0.0, 0.5])

observed_by_outcome = {
    outcome: float(frequency) for outcome, frequency in zip(outcomes, observed)
}
print("Observed frequencies:", observed_by_outcome)

figure, axis = plt.subplots(figsize=(7, 4))
positions = np.arange(len(outcomes))
axis.bar(positions, observed, width=0.65, label="seeded simulation")
axis.scatter(positions, ideal, color="black", marker="_", s=350, label="ideal")
axis.set(
    xticks=positions,
    xticklabels=outcomes,
    xlabel="Measured bitstring",
    ylabel="Frequency",
    ylim=(0, 0.6),
    title="Bell-state measurement frequencies",
)
axis.legend()
figure.tight_layout()
plt.show()
```

<!-- tutorial-result-start:cell-7 -->
!!! example "运行结果"

    ```text
    Observed frequencies: {'00': 0.502, '01': 0.0, '10': 0.0, '11': 0.498}
    ```

    ![固定随机种子的贝尔态测量频率与理想分布对比](../assets/generated/tutorials/bell-state-01.png)

<!-- tutorial-result-end:cell-7 -->

精确态矢量与采样直方图回答的是不同的问题。态矢量描述观测之前的相干量子态，其中包含振幅和相对相位。计数描述测量后的经典输出，其精度受 `shots` 限制。增加采样次数可缩小统计波动，但不能显露相位信息；以相位为目标的实验需要使用不同的测量基。

接下来，可以尝试将 Hadamard 门改为 `X`、删除受控 X 门，或使用用户指南中的噪声工具。可下载的 Python 源码是规范版本，因此无需从静态截图中复制代码就能探索每种变体。
