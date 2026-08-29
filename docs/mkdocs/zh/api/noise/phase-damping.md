
# PhaseDamping


[`PhaseDamping`][fatqat.noise.PhaseDamping] 在不转移布居的情况下消除相干性。其 `p` 模拟器通道与 `rate`／`t_phi` 仿真器 Lindblad 算符采用不同的多能级约定，因此应根据目标后端选择所需形式，而不要依赖隐式转换。

## 选择形式


必须且只能提供一个关键字：

**参数化形式**

| 参数 | 含义 | 约束 |
| --- | --- | --- |
| `p` | 一次模拟器通道应用的完全退相干权重 | `[0, 1]` 内的有限实数 |
| `rate` | 缩放局部退相干 Lindblad 算符的速率 | 逆时间单位下的有限非负实数 |
| `t_phi` | 纯退相干时间，归一化为 `rate = 1 / t_phi` | 以后端时间单位计的有限正实数 |

`t_phi` 是提供速率的便捷方式。对象会存储 `rate = 1 / t_phi`，而不是把 `t_phi` 作为单独值保存。

## 模拟器


对于维度为 $d$ 的目标，概率模式实现

$$
\mathcal{E}_p(\rho)
= (1-p)\rho + p\,\operatorname{diag}(\operatorname{diag}(\rho)).
$$

所有布居保持不变，每个非对角元素都乘以 $1-p$，与其两个能级之间的间隔无关。该通道作用于一个选定操作数。

## 脉冲仿真器


`rate` 和 `t_phi` 形式使用局部 Lindblad 算符

$$
L=\sqrt{2r}\,\operatorname{diag}(0,1,\ldots,d-1).
$$

它对能级 j 与 k 之间相干性的作用为

$$
\rho_{jk}(t)=e^{-r(j-k)^2t}\rho_{jk}(0).
$$

各内置仿真器接受的作用域和实现映射要求参阅[脉冲仿真器](backend-support.md#noise-emulator-support)。

## 两种形式之间的转换


[`PhaseDamping.as_probability`][fatqat.noise.PhaseDamping.as_probability] 使用 $p=1-e^{-rt}$，[`PhaseDamping.as_rate`][fatqat.noise.PhaseDamping.as_rate] 使用 $r=-\log(1-p)/t$。这些关系与量子比特通道和相邻能级相干性相符。对于更大系统，两种形式并不相同：模拟器通道会均匀衰减所有相干性，而 Lindblad 演化使衰减按 $(j-k)^2$ 缩放。

持续时间必须有限且非负。概率 1 不对应有限速率，非零概率在零持续时间下也无法转换。

## API


::: fatqat.noise.PhaseDamping
    options:
      members:
        - "as_probability"
        - "as_rate"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
