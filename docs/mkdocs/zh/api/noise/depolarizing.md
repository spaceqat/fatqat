
# Depolarizing


[`Depolarizing`][fatqat.noise.Depolarizing] 描述与最大混合态的均匀混合。模拟器通道使用 `p`，兼容脉冲仿真器上的 Lindblad 算符使用 `rate`。后端不会在两种形式之间自动转换。

## 选择形式


**参数化形式**

| 形式 | 含义 | 作用对象 | 典型后端 |
| --- | --- | --- | --- |
| `p` | 一次通道应用中完全退极化的权重，范围 `[0, 1]` | 选中的操作数 | 矩阵模拟器 |
| `rate` | 缩放局部退极化 Lindblad 算符的有限非负速率 | 一个子系统 | 支持 Lindblad 的脉冲仿真器 |

必须且只能提供一种形式。速率单位是所选后端时间单位的倒数。例如，持续时间以微秒计的后端会把速率解释为每微秒。

## 模拟器


对于合并维度为 $d$ 的选定操作数，概率模式为

$$
\mathcal{E}_p(\rho) = (1-p)\rho + p\frac{I_d}{d}.
$$

该通道联合作用于所有选定操作数。因此，把 `Depolarizing(p=...)` 应用于两个量子比特时使用 $d=4$，而不是创建两个相互独立的单量子比特通道。若需独立局部噪声，请使用 `target_positions` 或分别注册。

这里的 `p` 是完全退极化的权重，而不是选择非恒等误差的概率。对于一个量子比特，分配给非恒等分支的总概率为 $3p/4$。

在受支持的方法上，`statevector` 每次应用采样一个 Kraus 分支，`density_matrix` 应用精确的 Kraus 和，`superop` 构造完整通道。限制条件参阅[模拟器](backend-support.md#noise-simulator-support)。

## 脉冲仿真器


对于局部维度 $d$，速率模式经过归一化，其 Lindblad 生成元为

$$
\mathcal{L}_r(\rho)
= r\left(\operatorname{Tr}(\rho)\frac{I_d}{d}-\rho\right).
$$

经过持续时间 $t$ 后，对应概率参数为

$$
p(t)=1-e^{-rt}.
$$

已知持续时间时，[`Depolarizing.as_probability`][fatqat.noise.Depolarizing.as_probability] 和 [`Depolarizing.as_rate`][fatqat.noise.Depolarizing.as_rate] 会执行此参数转换。持续时间必须有限且非负。概率 1 不对应有限速率，非零概率在零持续时间下也无法转换。

速率模式要求已注册 Lindblad 实现。内置及自定义映射的可用性参阅[脉冲仿真器](backend-support.md#noise-emulator-support)。

## API


::: fatqat.noise.Depolarizing
    options:
      members:
        - "as_probability"
        - "as_rate"
        - "num_subsystems"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
