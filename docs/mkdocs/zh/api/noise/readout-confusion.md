<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# ReadoutConfusion


[`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] 是在物理测量后应用于报告数字的经典通道。它会改变计数和前馈输入，但不会改变测量后的量子状态。

## 矩阵约定


对于混淆矩阵 C，

$$
C_{r,t}=P(\text{reported}=r\mid\text{true}=t).
$$

行表示报告数字，列表示真实数字。因此，每一列之和必须为 1。量子比特矩阵

$$
C=\begin{pmatrix}
    0.98 & 0.04\\
    0.02 & 0.96
  \end{pmatrix}
$$

表示真实值 0 以 0.02 的概率报告为 1，而真实值 1 以 0.04 的概率报告为 0。

输入可以是任何可转换为浮点数的类数组值。它必须产生边长至少为 2 的有限方阵，各元素位于 `[0, 1]`，且各列之和在数值容差内为 1。矩阵大小还必须与后端报告数字的维度匹配。

FATQAT 会把矩阵转换为浮点数，并存储自己的只读副本。之后修改输入数组不会影响噪声对象。

## 测量报告方式


后端先采样真实物理结果并使量子状态坍缩，再从对应矩阵列中采样报告数字。报告值写入经典内存，因此后续前馈会看到混淆后的值。再次使用被测子系统时，演化仍从其真实坍缩状态开始。

## 作用位置


读出混淆总是在 [`NoiseModel`][fatqat.NoiseModel] 的测量阶段应用：

```python
noise.add(confusion)  # Every measured operand.

# Alternatively, target the "q0" device label on a transmon model.
targeted_noise = fq.NoiseModel()
targeted_noise.add(confusion, targets="q0")
```

省略 `targets` 会影响每个被测操作数，也可以传入一个量子 [`RegisterRef`][fatqat.RegisterRef] 或设备标签。不支持多操作数相关读出。不要传入 `operation` 或 `target_positions`，即使其值为 `None` 也不行。

通用注册不能与定向注册共存，同一目标也不能重复注册。如果逻辑注册和设备标签注册选择同一个操作数，程序使用具体布局运行时会拒绝二者。

## 模拟器


模拟器要求矩阵边长与被测子系统报告数字的维度一致。空位点擦除会绕过混淆，因为没有测得物理数字。后端和维度详情参阅[模拟器](backend-support.md#noise-simulator-support)。

## 脉冲仿真器


读出混淆仍是经典报告步骤，不由 Lindblad 算符表示。各仿真器的报告数字维度和物理能级映射参阅[脉冲仿真器](backend-support.md#noise-emulator-support)。

## API


::: fatqat.noise.ReadoutConfusion
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
