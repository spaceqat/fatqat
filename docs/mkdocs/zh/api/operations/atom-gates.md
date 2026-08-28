<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 原子阵列操作


`Put` 管理原子占据；`Pair` 和 `Unpair` 管理连接关系。它们不是酉矩阵门。使用 [`fatqat.Program.add`][fatqat.Program.add] 添加这些操作。内置后端中只有 [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] 实现了它们。其他矩阵和脉冲后端会引发 [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]。

**原子阵列操作**

| 值 | 标量目标 | 效果 | 条件 | 附加噪声 |
| --- | --- | --- | --- | --- |
| [`Put`][fatqat.operations.Put] | 一个或多个 | 将 `\|0>` 装载到每个空位点；已占据位点保持不变。 | 允许。 | 仅 [`Loss`][fatqat.noise.Loss]，在每个启用的 `Put` 操作之后。 |
| [`Pair`][fatqat.operations.Pair] | 恰好两个 | 添加二者间的无向连接边；重复配对不产生变化。 | 拒绝。 | [`Loss`][fatqat.noise.Loss] 或受支持的有限通道。 |
| [`Unpair`][fatqat.operations.Unpair] | 恰好两个 | 移除二者间的边；移除不存在的边不产生变化。 | 拒绝。 | [`Loss`][fatqat.noise.Loss] 或受支持的有限通道。 |

如果程序包含 `Put`，每个 shot 开始时所有已声明位点都为空。位点只在 `Put` 运行时被占据，之后的 `Put` 可以重新装载丢失的原子。附加到 `Put` 的 [`Loss`][fatqat.noise.Loss] 声明与该操作共享条件，并会在每个条件通过的匹配 `Put` 操作之后运行；即使位点已经被占据、`Put` 本身没有产生变化也同样如此。

`Pair` 和 `Unpair` 更新供后续受支持门使用的连接关系；它们不会改变量子状态，也不会让原本不受支持的门变得可用。在内置原子阵列配置中，[`CZ`][fatqat.operations.CZ] 是原生门，要求目标之间当前已配对。程序运行时，如果这两种指令带有条件，原子后端会以 [`BackendValidationError`][fatqat.errors.BackendValidationError] 拒绝。

::: fatqat.operations.Put
    options:
      show_attribute_values: false

::: fatqat.operations.Pair
    options:
      show_attribute_values: false

::: fatqat.operations.Unpair
    options:
      show_attribute_values: false
