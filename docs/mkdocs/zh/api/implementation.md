
# 矩阵实现


使用 [`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap] 向模拟器添加自定义矩阵操作，或按物理目标选择门规则。构造后端时以 `implementation_map=` 传入该映射。内置门无需自定义映射即可工作。

## 选择映射


[`default_matrix_implementation_map`][fatqat.implementation.default_matrix_implementation_map] 会返回一个包含 FATQAT 内置矩阵门的新映射。修改它不会影响其他映射或之后的调用。如果后端只应支持由你添加的规则，请直接构造 [`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap]。

## 选择规则


[`MatrixImplementationMap.add`][fatqat.implementation.MatrixImplementationMap.add] 接受操作实例或类，以及以下三种规则形式之一：

**矩阵规则形式**

| 形式 | 适用场景 | 传给 `add` 的值 |
| --- | --- | --- |
| 二维 NumPy 数组 | 一个常量矩阵 | 数组；FATQAT 会复制它，因此之后的修改不影响规则 |
| 可调用对象 | 依赖操作参数或目标维度的矩阵 | 默认调用 `rule(op)`。若可调用对象接受 `targets=` 关键字或 `**kwargs`，FATQAT 会调用 `rule(op, targets=targets)`。 |
| [`MatrixImplementation`][fatqat.implementation.MatrixImplementation] | 已配置或有状态的规则对象 | 重写 `__call__(op, *, targets)`。 |

`op` 是实际应用的操作值，因此包含旋转角等参数。`targets` 按操作数顺序包含标量程序 [`RegisterRef`][fatqat.RegisterRef] 对象；矩阵依赖子系统维度时，规则可检查 `target.register.dim`。

## 局部基顺序


矩阵因子遵循传给 [`add`][fatqat.Program.add] 的目标元组顺序。第一个目标是最高有效局部因子，最后一个目标变化最快。对于维度 `(d0, d1, ..., dk)` 和基数字 `(b0, b1, ..., bk)`，展平后的局部索引为 `b0 * d1 * ... * dk + b1 * d2 * ... * dk + ... + bk`。

因此，对两个量子比特而言，目标 `(q0, q1)` 使用局部基 `|00>`、`|01>`、`|10>`、`|11>`。受控操作把控制目标列在作用目标之前。此局部约定独立于完整系统的显示顺序和结果比特顺序。

## 目标专用规则


向 `add` 传入 `device_operands=`，可使规则仅用于后端定义的物理标签的精确有序元组。这些标签不是程序寄存器引用。一个操作类别只能采用一种注册模式：

- 不带 `device_operands` 添加的*统一*规则匹配所有元数正确的物理元组。
- *设备专用*规则只匹配其显式注册的元组。同一操作可以添加多个元组。

在两种模式间切换某个操作类别前，请调用 [`remove`][fatqat.implementation.MatrixImplementationMap.remove]。

## 验证时机


`add` 会立即以 `TypeError` 或 `ValueError` 拒绝无效注册。可调用对象在后端准备程序时运行。规则抛出异常会产生 [`MatrixImplementationError`][fatqat.errors.MatrixImplementationError]，数组目标形状错误会产生 [`BackendValidationError`][fatqat.errors.BackendValidationError]，缺少规则会产生 [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]。这些错误都发生在 `run` 返回 [`Job`][fatqat.Job] 之前。

## 参考


### 数据 `fatqat.implementation.DeviceOperands` { #fatqat.implementation.DeviceOperands }

由后端定义的可哈希物理标签有序元组的别名。

::: fatqat.implementation.default_matrix_implementation_map

::: fatqat.implementation.MatrixImplementationMap
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.implementation.MatrixImplementation
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.implementation.FixedMatrix
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
