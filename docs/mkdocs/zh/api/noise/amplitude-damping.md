<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# AmplitudeDamping


[`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] 描述向基态方向的相邻能级衰减。对于模拟器通道使用 `p`，对于兼容仿真器上的局部 Lindblad 算符使用 `rate`；每个跃迁 $|k\rangle\rightarrow|k-1\rangle$ 对应一个值。

## 选择概率或速率


`p` 和 `rate` 必须且只能提供一个。二者都接受单个实数或非空可迭代对象；FATQAT 会把这些值存为元组。对于 $d$ 能级目标，必须恰好传入 $d-1$ 个值：

**条目顺序**

| 条目 | 跃迁 | 约束 |
| --- | --- | --- |
| `value[0]` | $\|1\rangle\rightarrow\|0\rangle$ | `[0, 1]` 内的概率或非负速率 |
| `value[1]` | $\|2\rangle\rightarrow\|1\rangle$ | 仅当 $d\geq3$ 时需要 |
| `value[d - 2]` | $\|d-1\rangle\rightarrow\|d-2\rangle$ | 维度 $d$ 的最后一个值 |

因此，标量只适用于二能级目标。程序运行、目标维度已知时，FATQAT 会检查值的数量。

## 模拟器


对于概率 $p_1,\ldots,p_{d-1}$，模拟器使用

$$
\begin{aligned}
K_0 &= |0\rangle\!\langle0|
       + \sum_{k=1}^{d-1}\sqrt{1-p_k}|k\rangle\!\langle k|,\\
K_1 &= \sum_{k=1}^{d-1}\sqrt{p_k}|k-1\rangle\!\langle k|.
\end{aligned}
$$

一次应用最多把布居向下移动一个相邻能级。该通道作用于一个选定操作数；可用 `target_positions` 选择多操作数门中的某个操作数。内置支持情况参阅[模拟器](backend-support.md#noise-simulator-support)。

## 脉冲仿真器


对于速率 $r_k$，兼容脉冲后端使用局部 Lindblad 算符

$$
L = \sum_{k=1}^{d-1}\sqrt{r_k}|k-1\rangle\!\langle k|.
$$

速率必须有限且非负，单位为后端时间单位的倒数。各内置仿真器的局部维度、接受的作用域和实现映射要求参阅[脉冲仿真器](backend-support.md#noise-emulator-support)。

## 两种形式之间的转换


转换方法对每个跃迁独立应用

$$
p_k(t)=1-e^{-r_k t}
$$

对于二能级系统，这是精确关系。对于 $d>2$，Lindblad 演化可以在一个时间区间内发生多次相邻跃迁，而一次模拟器通道应用最多把布居下移一个能级。因此，应把返回元组视为参数转换，而不是该区间内精确的多能级演化。

持续时间必须有限且非负。概率 1 不对应有限速率，非零概率在零持续时间下也不能转换为有限速率。

## API


::: fatqat.noise.AmplitudeDamping
    options:
      members:
        - "as_probability"
        - "as_rate"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
