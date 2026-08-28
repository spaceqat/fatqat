<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 测量与结构操作


## 测量与重置


测量将计算基结果写入经典存储。重置会把目标恢复到 `|0>`，但不产生输出。

使用 [`fatqat.Program.measure`][fatqat.Program.measure] 或 [`fatqat.Program.measure_all`][fatqat.Program.measure_all] 创建测量。分组测量按元组位置配对量子目标和经典输出，每一对必须具有相同的局部维度。测量通过这些方法创建，而不是通过 [`add`][fatqat.Program.add]，也不能携带后者的 `condition=` 参数。

允许重复的目标和输出，各对按元组顺序处理。重复目标会报告其已坍缩的结果，并对每一对独立应用读出噪声。重复经典输出时，最后一次写入生效。[`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] 只改变报告的数字，不改变坍缩后的物理结果。

::: fatqat.operations.Measurement
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

使用 [`fatqat.Program.add`][fatqat.Program.add] 添加 [`Reset`][fatqat.operations.Reset]。它接受一个或多个互不相同的标量目标，并且在后端支持前馈时可以携带条件。空目标元组、重复目标和 [`RegisterView`](../registers.md#fatqat.RegisterView) 都会被拒绝。重置是非酉操作。对于纠缠目标，状态向量运行会采样一个重置分支，而密度矩阵运行会直接表示所得混合态。它没有附加噪声实现，因此在 [`fatqat.NoiseModel.add`][fatqat.NoiseModel.add] 中将其用作 `operation=` 选择器会引发 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)。

::: fatqat.operations.Reset
    options:
      show_attribute_values: false

## 编译器屏障


[`Barrier`][fatqat.operations.Barrier] 是编译和调度标记，不是改变状态的操作或噪声边界。使用 [`fatqat.Program.add`][fatqat.Program.add] 添加，并传入一个或多个互不相同的标量目标。空目标元组、重复目标和 [`RegisterView`](../registers.md#fatqat.RegisterView) 都会被拒绝。

内置模拟器会忽略屏障，包括 [`add`][fatqat.Program.add] 记录的任何条件，因此屏障不会改变状态或计数。屏障不能绑定噪声：在 [`fatqat.NoiseModel.add`][fatqat.NoiseModel.add] 中将其用作 `operation=` 选择器会引发 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)。

::: fatqat.operations.Barrier
    options:
      show_attribute_values: false
