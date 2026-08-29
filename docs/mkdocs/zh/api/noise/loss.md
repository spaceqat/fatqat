
# Loss


[`Loss`][fatqat.noise.Loss] 从可感知占据状态的模拟中移除物理载体。它不同于振幅阻尼：振幅阻尼使子系统保留在建模的 Hilbert 空间中，而损失会把载体标记为缺失，并丢弃其量子关联。

## 损失的作用位置


`p` 是每个当前存在的选定载体的损失概率。匹配操作之后，FATQAT 会独立采样各载体。作用于整个操作的注册对每个操作数使用相同概率；可用 `target_positions` 将损失限制到特定操作数。

[`Loss`][fatqat.noise.Loss] 只能附加到匹配操作，不能注册为背景噪声。匹配操作上的条件也控制其损失：如果跳过该操作，其附加损失也会跳过。

## 可感知占据状态的模拟器


在可感知占据状态的模拟器上，任意匹配的损失注册都会启用原子生命周期：

* 每个位点初始为空；
* `Put` 向空位点装载一个新的 `|0>` 原子；
* 一次损失命中会移除存在的原子及其关联；
* 之后需要该原子的门在对应 shot 中不产生作用；
* 之后的 `Put` 可以重新填充位点；
* 测量空位点会报告擦除数字 `2`。

擦除会绕过 [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion]，因为没有已占据量子比特产生物理读出数字。把损失附加到 `Put` 时，会在装载后采样，可用于模拟装载失败或装载后立即损失。`Put` 不接受其他噪声类型。

启用生命周期的是注册本身，而不是采样到的损失事件。因此，即使匹配 `Loss(p=0)`，所有位点仍会从空状态开始，并要求显式执行 `Put`。

每个 shot 都有独立的占据状态。`statevector` 和 `density_matrix` 支持此生命周期，`unitary` 和 `superop` 不支持。由于最终状态取决于采样得到的损失历史，请求最终状态时必须只运行一个 shot。

内置的可感知占据状态后端及其方法限制参阅[模拟器](backend-support.md#noise-simulator-support)。

## API


::: fatqat.noise.Loss
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
