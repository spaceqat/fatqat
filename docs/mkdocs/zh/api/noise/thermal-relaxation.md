
# ThermalRelaxation


[`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] 对仿真器上的零温 T1 和 T2 弛豫进行建模。它将向下布居衰减与复现 T2 所需的额外纯退相干结合。对于量子比特模拟器，请在已知持续时间下用 [`ThermalRelaxation.as_channels`][fatqat.noise.ThermalRelaxation.as_channels] 显式转换。

## 时间与速率


`t1` 和 `t2` 是采用同一单位的有限正值；它们本身没有固有单位。传给 [`ThermalRelaxation.as_channels`][fatqat.noise.ThermalRelaxation.as_channels] 的持续时间必须使用相同单位。向仿真器注册噪声时，请使用仿真器模型的时间单位。物理一致性要求

$$
T_2 \leq 2T_1.
$$

导出的速率为

$$
\gamma_1 = \frac{1}{T_1}, \qquad
\gamma_\phi = \frac{1}{T_2}-\frac{1}{2T_1}.
$$

[`ThermalRelaxation.amplitude_rate`][fatqat.noise.ThermalRelaxation.amplitude_rate] 返回 $\gamma_1$，[`ThermalRelaxation.pure_dephasing_rate`][fatqat.noise.ThermalRelaxation.pure_dephasing_rate] 返回 $\gamma_\phi$。在 T2 边界下，后者非负，并在 $T_2=2T_1$ 时变为零。

## 脉冲仿真器


对于 $d$ 能级脉冲模型，仿真器使用局部 Lindblad 算符

$$
\begin{aligned}
L_1 &= \sum_{k=1}^{d-1}\sqrt{\frac{k}{T_1}}
       |k-1\rangle\!\langle k|,\\
L_\phi &= \sqrt{2\gamma_\phi}\,
          \operatorname{diag}(0,1,\ldots,d-1).
\end{aligned}
$$

当 $\gamma_\phi=0$ 时省略第二个算符。这是一个局部零温模型：它包含向下弛豫，但不包含热激发或平衡布居参数。

各内置仿真器接受的作用域和实现映射要求参阅[脉冲仿真器](backend-support.md#noise-emulator-support)。

## 模拟器转换


[`ThermalRelaxation.as_channels`][fatqat.noise.ThermalRelaxation.as_channels] 对持续时间 $t$ 返回一个有序对：

$$
\begin{aligned}
p_1(t) &= 1-e^{-t/T_1},\\
p_\phi(t) &= 1-e^{-\gamma_\phi t}.
\end{aligned}
$$

先应用返回的 [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping]，再应用 [`PhaseDamping`][fatqat.noise.PhaseDamping]。对于量子比特，二者组合给出布居衰减 $e^{-t/T_1}$ 和相干性衰减 $e^{-t/T_2}$。

此转换适用于量子比特。它返回一个振幅概率，而更高维的振幅阻尼通道需要 $d-1$ 个值。`ThermalRelaxation` 本身不是模拟器通道。模拟器支持情况参阅[模拟器](backend-support.md#noise-simulator-support)。

## API


::: fatqat.noise.ThermalRelaxation
    options:
      members:
        - "amplitude_rate"
        - "pure_dephasing_rate"
        - "as_channels"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
