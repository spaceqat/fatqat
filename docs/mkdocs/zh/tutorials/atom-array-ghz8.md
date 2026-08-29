---
title: "将八个原子纠缠为 GHZ 态"
description: "利用动态 Pair 和 Unpair 操作构建八原子 GHZ 态，再检验其关联与相干相位。"
---
<!-- 中文译文人工维护；运行结果由 docs/mkdocs/tools/convert_tutorials.py 从规范源码同步。 -->

# 将八个原子纠缠为 GHZ 态

<div class="grid cards" markdown>

-   :material-map-marker-path: **学习路径**

    中性原子物理

-   :material-language-python: **可执行源码**

    [下载 `plot_atom_array_ghz8.py`](../downloads/tutorials/plot_atom_array_ghz8.py){ download }

</div>

本教程在 fatqat 的中性原子执行目标 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 上构建八原子 Greenberger-Horne-Zeilinger（GHZ）态。它比两量子比特贝尔态教程更进一步：该状态横跨八个而非两个原子，电路运行在双比特门连接可以在*电路执行途中重配置*的后端上，而不是使用固定连接。

GHZ 态将贝尔态推广到 $n$ 个量子比特：

$$
|\mathrm{GHZ}_8\rangle
  = \frac{|00000000\rangle + |11111111\rangle}{\sqrt{2}}.
$$

与贝尔态一样，它无法分解为相互独立的单原子态。测量全部八个原子时，*只会*得到 `00000000` 或 `11111111`，两者概率均为二分之一，因此任一原子的结果都与其余原子完全关联。不过，仅有这种关联并不足以证明量子性：一次同时设定八个比特的经典抛硬币也能产生同样的结果。GHZ 态的量子特性在于两条分支保持在具有确定相对相位的*相干叠加*中。我们将先用采样计数确认关联，然后用精确期望值确认相干性。

中性原子方案的特别之处在于连接性。在该后端上，`CZ` 只能作用于当前已*成对*的原子。电路运行时，[`Pair`][fatqat.operations.Pair] / [`Unpair`][fatqat.operations.Unpair] 操作会更新这个配对状态。在物理上，`Pair` 表示“将这两个原子移入同一纠缠区域”，`Unpair` 则表示“再次将它们分开”。正是这种原子搬运赋予可重配置中性原子处理器可编程连接性。我们将利用它纠缠在任何固定一维布局中都不会相邻的原子。

!!! info "基于源码的教程"

    说明文字是对版本库中教程源码的人工中文翻译，页面中的可执行单元保留规范源码。转换脚本从同一源码捕获运行结果；其中的英文标签来自源码的打印语句，保留原样以便核对。页面不显示仅用于 Sphinx-Gallery 验证的代码段。下载并直接运行 Python 文件即可复现图形与标准输出。

## 导入与显示设置

[`fatqat.Program`][fatqat.Program] 是与后端无关的电路描述。量子门定义位于 [`fatqat.operations`](../api/operations.md)。NumPy 用于整理期望值，运行规范源码时，Matplotlib 会生成图形。

```python title="Python 单元 1"
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

np.set_printoptions(precision=3, suppress=True)

NUM_ATOMS = 8
```

## 原生门集

原子阵列后端只接受其原生门：[`RX`][fatqat.operations.RX]、[`RY`][fatqat.operations.RY]、[`RZ`][fatqat.operations.RZ] 和 [`CZ`][fatqat.operations.CZ]。`H` 和 `CX` 这样的便利门在这里不是原生门，因此我们手工编译它们——这正是硬件感知转译器会做的事。在忽略无关紧要的全局相位后，Hadamard 门等于先施加 $R_Z(\pi)$，再施加 $R_Y(\pi/2)$；受控 X 门则是由 Hadamard 门共轭的 `CZ`：

$$
\mathrm{CX}(c \to t) = H_t \, \mathrm{CZ}(c, t) \, H_t.
$$

将它们写成小型辅助函数，可使下方的电路构建代码保持清晰。

```python title="Python 单元 2"
def native_h(program: fq.Program, target: int) -> None:
    """Hadamard in the native gate set: ``RZ(pi)`` then ``RY(pi/2)``."""
    program.add(ops.RZ(np.pi), target)
    program.add(ops.RY(np.pi / 2), target)


def native_cx(program: fq.Program, control: int, target: int) -> None:
    """``CX(control -> target)`` as ``H(target) CZ H(target)``.

    The ``CZ`` in the middle is valid only while ``control`` and ``target`` are
    currently paired; otherwise the backend raises
    :class:`~fatqat.errors.BackendValidationError`.
    """
    native_h(program, target)
    program.add(ops.CZ, (control, target))
    native_h(program, target)
```

## 对数深度的纠缠树

我们可以用七个受控 X 门组成的链来扩展 GHZ 态，让原子 `0` 依次连接每个相邻原子。本例改用*二叉树*，只需三层就能覆盖全部八个原子，而且同一层内的量子门可以在相互独立的原子对上并行运行：

```text
第 1 层:  (0,4)                             1 个 CX  -- 为树设置种子
第 2 层:  (0,2) || (4,6)                    2 个 CX  并行
第 3 层:  (0,1) || (2,3) || (4,5) || (6,7)  4 个 CX  并行
```

同一层内的各对原子互不相交，因此这些受控 X 门作用于不同原子，硬件可同时执行它们。

```python title="Python 单元 3"
CX_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 4),),
    ((0, 2), (4, 6)),
    ((0, 1), (2, 3), (4, 5), (6, 7)),
)

for layer_index, layer in enumerate(CX_LAYERS, start=1):
    print(f"layer {layer_index}: " + " || ".join(f"CX{pair}" for pair in layer))
```

<!-- tutorial-result-start:cell-3 -->
!!! example "运行结果"

    ```text
    layer 1: CX(0, 4)
    layer 2: CX(0, 2) || CX(4, 6)
    layer 3: CX(0, 1) || CX(2, 3) || CX(4, 5) || CX(6, 7)
    ```

<!-- tutorial-result-end:cell-3 -->

## 为什么层与层之间必须搬运原子对

该树有意耦合相距较远的原子。第 1 层纠缠原子 `0` 和 `4`；第 2 层需要将 `0` 与 `2`、`4` 与 `6` 分别相连。任何固定的最近邻布局都无法同时提供所有这些邻接关系。

这正是可重配置连接性的价值所在。在每层量子门之前，我们先对层内原子执行 [`Pair`][fatqat.operations.Pair]——将每对原子搬运到同一纠缠区域——并在该层结束后立即执行 [`Unpair`][fatqat.operations.Unpair]，释放原子以便下一层重新分组。如果跳过这种重排，第 2 层和第 3 层的 `CZ` 将指向未成对原子，后端会以 [`BackendValidationError`][fatqat.errors.BackendValidationError] 拒绝该程序。原子搬运并非算法的附带细节；它*就是*算法的布线方式。

```python title="Python 单元 4"
def build_ghz8_program(*, measure: bool = True) -> fq.Program:
    """Assemble the eight-atom GHZ program.

    With ``measure=True`` every atom is read into a classical bit at the end,
    which is what the counts experiment needs. With ``measure=False`` the
    program has no classical register, leaving the coherent final state for the
    :class:`~fatqat.Estimator` to interrogate.
    """
    program = fq.Program(NUM_ATOMS, NUM_ATOMS if measure else 0)

    # Sites start empty; load one |0> atom into each of the eight traps.
    program.add(ops.Put, tuple(range(NUM_ATOMS)))

    # Seed the tree: put atom 0 into |+>, the root the branches grow from.
    native_h(program, 0)

    for layer in CX_LAYERS:
        for pair in layer:  # transport the layer's atoms together
            program.add(ops.Pair, pair)
        for control, target in layer:  # one parallel layer of CX = H CZ H
            native_cx(program, control, target)
        program.add(ops.Barrier, tuple(range(NUM_ATOMS)))  # visual layer marker
        for pair in layer:  # move the pairs apart again
            program.add(ops.Unpair, pair)

    if measure:
        program.measure_all()
    return program


ghz_program = build_ghz8_program()
```

## 对八原子实验进行采样

我们在设置为八个陷阱位点的 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 上运行含测量的程序。固定随机种子可使本页的数值在多次教程运行中保持稳定；若需要独立的实验样本，可将其省略。由于电路无噪声，2000 次采样中的每一次都落在两个 GHZ 分支之一，不会出现其他比特串。

```python title="Python 单元 5"
shots = 2_000
backend = fq.simulator.AtomArraySimulator(num_sites=NUM_ATOMS)
counts = (
    backend.run(
        ghz_program,
        shots=shots,
        simulation_config={"seed": 7},
    )
    .result()
    .get_counts()
)

print("Counts:", counts)
```

<!-- tutorial-result-start:cell-5 -->
!!! example "运行结果"

    ```text
    Counts: {'00000000': 1017, '11111111': 983}
    ```

<!-- tutorial-result-end:cell-5 -->

只会出现 `00000000` 和 `11111111`，两者各占约一半采样。下方柱形图将观测频率与理想值 $1/2$ 对照展示；有限采样使每个柱子稍微偏离该参考线，而所有中间比特串的频率仍为零。

```python title="Python 单元 6"
all_zero, all_one = "0" * NUM_ATOMS, "1" * NUM_ATOMS
observed = np.array([counts.get(all_zero, 0), counts.get(all_one, 0)]) / shots

figure, axis = plt.subplots(figsize=(7, 4))
positions = np.arange(2)
axis.bar(positions, observed, width=0.55, label="seeded simulation")
axis.scatter(positions, [0.5, 0.5], color="black", marker="_", s=350, label="ideal")
axis.set(
    xticks=positions,
    xticklabels=(all_zero, all_one),
    xlabel="Measured bitstring",
    ylabel="Frequency",
    ylim=(0, 0.6),
    title="GHZ$_8$ measurement frequencies",
)
axis.legend()
figure.tight_layout()
plt.show()
```

<!-- tutorial-result-start:cell-6 -->
!!! example "运行结果"

    ![八原子 GHZ 态的测量频率](../assets/generated/tutorials/atom-array-ghz8-01.png)

<!-- tutorial-result-end:cell-6 -->

## 关联并不等于相干

计数证明八个原子完全*关联*，但“全零”与“全一”之间等概率切换的经典混合态也会给出完全相同的直方图。为了看到量子相干性，我们对混合态与叠加态结果不同的可观测量进行测量，在未测量态上使用精确 [`Estimator`][fatqat.Estimator]。

两类见证量可以锁定 $|\mathrm{GHZ}_8\rangle$。每个相邻奇偶性 $\langle Z_i Z_{i+1}\rangle = 1$ 表明相邻原子始终一致，经典混合态也满足这一条件。全局 $X$ 奇偶性 $\langle X_0 X_1 \cdots X_7\rangle = 1$ 则是区分标志：只有相干叠加态会取 $+1$，混合态的平均值则为 $0$。同时观察到两类结果，才能确认真正的 GHZ 态。

```python title="Python 单元 7"
zz_observables = []
for i in range(NUM_ATOMS - 1):
    label = ["I"] * NUM_ATOMS
    label[i] = label[i + 1] = "Z"
    zz_observables.append(fq.Observable([("".join(label), 1.0)]))
x_parity = fq.Observable([("X" * NUM_ATOMS, 1.0)])

estimator = fq.Estimator(fq.simulator.AtomArraySimulator(num_sites=NUM_ATOMS))
values = (
    estimator.run(
        build_ghz8_program(measure=False),
        zz_observables + [x_parity],
    )
    .result()
    .get_expectation()
)

for i, value in enumerate(values[:-1]):
    print(f"<Z{i}Z{i + 1}> = {value:+.6f}")
print(f"<{'X' * NUM_ATOMS}> = {values[-1]:+.6f}")
```

<!-- tutorial-result-start:cell-7 -->
!!! example "运行结果"

    ```text
    <Z0Z1> = +1.000000
    <Z1Z2> = +1.000000
    <Z2Z3> = +1.000000
    <Z3Z4> = +1.000000
    <Z4Z5> = +1.000000
    <Z5Z6> = +1.000000
    <Z6Z7> = +1.000000
    <XXXXXXXX> = +1.000000
    ```

<!-- tutorial-result-end:cell-7 -->

每个见证量都返回 $+1$，包括全局 $X$ 奇偶性，因此该状态是相干 GHZ 叠加态，而非只能复现计数的经典混合态。

## 搬运原子的代价

可重配置连接性并非没有代价：每次搬运都存在原子逃离阱的风险。我们为 `Pair` 和 `Unpair` 操作附加 [`Loss`][fatqat.noise.Loss] 来建模这一现象：每次移动原子时，它都以一定概率被移出。只有这个中性原子后端能够建模原子丢失；通用模拟器会拒绝相同的噪声模型。

丢失的原子既不是 `|0>` 也不是 `|1>`：它已经*不存在*，读出时显示为擦除数字 `2`，与两种计算基结果都不同。因此，任何比特串中含 `2` 的采样都表明搬运过程中至少丢失了一个原子。

```python title="Python 单元 8"
noise = fq.NoiseModel()
noise.add(fq.noise.Loss(p=0.01), operation=ops.Pair)
noise.add(fq.noise.Loss(p=0.01), operation=ops.Unpair)

lossy_backend = fq.simulator.AtomArraySimulator(num_sites=NUM_ATOMS, noise=noise)
lossy_counts = (
    lossy_backend.run(
        ghz_program,
        shots=shots,
        simulation_config={"seed": 7},
    )
    .result()
    .get_counts()
)

lost_shots = sum(n for bitstring, n in lossy_counts.items() if "2" in bitstring)
print(f"{lost_shots}/{shots} shots lost at least one atom (a '2' in the readout)")
print("Most frequent outcomes under 1% loss per move:")
for bitstring, n in sorted(lossy_counts.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {bitstring}: {n}")
```

<!-- tutorial-result-start:cell-8 -->
!!! example "运行结果"

    ```text
    469/2000 shots lost at least one atom (a '2' in the readout)
    Most frequent outcomes under 1% loss per move:
      11111111: 799
      00000000: 732
      00020000: 48
      00000002: 48
      00000200: 30
      02000000: 29
    ```

<!-- tutorial-result-end:cell-8 -->

## 要点与下一步

本教程构建了真正的八原子 GHZ 态，通过采样计数验证其关联，通过精确期望值验证其相干性，并看到通过搬运原子重配置连接性如何既实现对数深度纠缠树，又引入真实的丢失通道。

接下来，可以通过添加第四层将树扩展到十六个原子，调高丢失概率以观察擦除如何增多，或有意删除某一层的 `Pair` / `Unpair` 调用，看看 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 如何以 [`BackendValidationError`][fatqat.errors.BackendValidationError] 拒绝第一个未配对的 `CZ`。在探索物理丢失前请恢复配对：未配对量子门是程序错误，原子丢失后跳过量子门则是每次采样中的物理效应。可下载的 Python 文件是规范的可执行源码，因此可直接在可运行代码中尝试这些变体。
