
# Result


一个 [`Result`][fatqat.Result] 仅包含单次运行所产生的字段。字段为可选项时，先检查 `available_data`，再使用对应访问器。访问不可用的字段会引发 [`ResultFieldUnavailableError`][fatqat.errors.ResultFieldUnavailableError]，而不是返回 `None`。

**结果字段**

| 字段 | 访问器 | 产生方式 |
| --- | --- | --- |
| `"counts"` | [`get_counts`][fatqat.Result.get_counts] 或 [`get_counts_as_tuples`][fatqat.Result.get_counts_as_tuples] | 启用测量计数的后端运行 |
| `"statevector"` | [`get_statevector`][fatqat.Result.get_statevector] | 启用最终状态输出的状态向量运行 |
| `"density_matrix"` | [`get_density_matrix`][fatqat.Result.get_density_matrix] | 启用最终状态输出的密度矩阵运行 |
| `"unitary"` | [`get_unitary`][fatqat.Result.get_unitary] | 启用最终状态输出的酉矩阵运行 |
| `"superop"` | [`get_superop`][fatqat.Result.get_superop] | 启用最终状态输出的超算符运行 |
| `"expectation"` 和 `"std"` | [`get_expectation`][fatqat.Result.get_expectation] 和 [`get_std`][fatqat.Result.get_std] | [`Estimator`][fatqat.Estimator] 运行 |
| 后端扩展名称 | [`get_data`][fatqat.Result.get_data] | 后端扩展 |

`"final_state"` 是请求名称，不是可用数据名称。产生的状态会使用上表中对应具体表示形式的名称。确定性运行默认启用最终状态输出。

## 顺序与可变值


[`get_counts`][fatqat.Result.get_counts] 返回一个新的显示字符串字典。编号最高的经典槽位在左，槽位 0 在右。如果任一经典维度不小于 10，则以逗号分隔多位结果以避免歧义。[`get_counts_as_tuples`][fatqat.Result.get_counts_as_tuples] 则把展平后的经典槽位 0 放在元组位置 0。

其他大多数访问器直接返回结果中存储的值。如需保留原始值，请在修改数组或字典前先复制。元数据会记录规范化后的 `simulation_config` 和 `result_config`。后端扩展可以添加字段；脉冲仿真器结果还包含通用求解器设置。若要复现一次物理运行，请将模型、排列、控制和应用元数据与结果一同保存。

对于每个完整状态或算符，`metadata["state_axes"]` 会按从最低有效到最高有效的顺序列出物理子系统。每个条目包含一个 `device_operand` 及其程序 `register_ref`；如果物理模型包含 Program 未寻址的子系统，则 `register_ref` 为 `None`。位置 0 是展平基索引中的最低有效子系统。对于局部维度 `dims`，位置 `q` 的位权为 `prod(dims[:q])`。密度矩阵的行和列使用相同的基顺序。

仅计数的运行会将所有从未被测量写入的已声明经典槽位补零，并发出标准 `UserWarning`。这通常表示缺少测量。

引导式结果解读工作流程参阅[从一次运行中获取答案](../guide/interpret-results.md)；上述约定是规范的状态轴和计数顺序契约。

## 详细参考


::: fatqat.Result
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
