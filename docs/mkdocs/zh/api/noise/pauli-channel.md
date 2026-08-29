
# PauliChannel


[`PauliChannel`][fatqat.noise.PauliChannel] 表示 Pauli 字符串误差的随机混合。它适合描述均匀 [`Depolarizing`][fatqat.noise.Depolarizing] 无法表达的有偏或相关量子比特噪声。

## 项


传入一个映射或由 `(string, probability)` 对组成的可迭代对象。每个字符串必须：

* 非空；
* 只包含大写 `I`、`X`、`Y` 和 `Z`；
* 与其他所有字符串宽度相同。

每个概率必须是 `int` 或 `float`，不能是 `bool`，并且必须有限且位于 `[0, 1]`。字符串重复会报错。非恒等项的概率之和可以小于 1；FATQAT 会把剩余权重分配给全恒等字符串。若显式提供恒等项，其值必须与该隐含值一致。允许微小的浮点舍入误差。

FATQAT 会消费输入，并将 `PauliChannel.terms` 存为不可变元组。恒等项位于最前，其余项保持输入顺序：

```python
import fatqat as fq

channel = fq.noise.PauliChannel({"X": 0.01, "Z": 0.02})
assert channel.terms == (("I", 0.97), ("X", 0.01), ("Z", 0.02))
```

## 模拟器


补齐恒等权重后，该通道为

$$
\mathcal{E}(\rho) = \sum_i p_i P_i \rho P_i.
$$

兼容模拟器会在匹配操作之后应用此通道。内置支持情况和作用域参阅[模拟器](backend-support.md#noise-simulator-support)。

## 目标顺序


字符串宽度决定目标数量，每个目标都必须是量子比特。第一个字符描述第一个目标，并构成最高有效张量因子。对于目标 `(q0, q1)`：

**双量子比特顺序**

| 字符串 | 第一个目标 `q0` | 第二个目标 `q1` |
| --- | --- | --- |
| `XI` | `X` | `I` |
| `IX` | `I` | `X` |

这种从左到右的约定与 Qiskit 显示 Pauli 字符串的顺序相反。程序运行时，FATQAT 会检查目标数量和量子比特维度。

## API


::: fatqat.noise.PauliChannel
    options:
      members:
        - "num_subsystems"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
