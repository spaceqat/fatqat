<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 可观测量与估计


使用 [`Estimator`][fatqat.Estimator] 从后端的最终状态计算一个或多个 [`Observable`][fatqat.Observable] 的值。程序必须没有测量、已完全绑定参数且仅含量子比特；后端必须返回状态向量或密度矩阵。

## 估计可观测量


```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))

estimator = fq.Estimator(fq.simulator.Simulator("SV"))
observable = fq.Observable([("ZZ", 1.0)])
result = estimator.run(program, observable).result()
expectation = result.get_expectation()
```

引导式工作流程参阅[从一次运行中获取答案](../guide/interpret-results.md)。

## 构造可观测量


稠密标签将量子比特 0 放在右侧，并接受 `I`、`X`、`Y` 和 `Z`。以下两种形式等价：

```python
fq.Observable([("ZZ", 1.5)])
fq.Observable(["ZZ"], coeffs=[1.5])
```

[`from_sparse`][fatqat.Observable.from_sparse] 会显式指定每个非恒等因子及其量子比特。它还支持 `ZERO` 和 `ONE` 投影算符：

```python
fq.Observable.from_sparse(
    [(["ONE", "Z"], (5, 3), 1.5)],
    num_qubits=6,
)
```

系数必须为实数。

## 精确结果与采样结果


传入一个可观测量时得到标量期望值和标准误差；传入列表或元组时，则按相同顺序得到数组。

`shots=0` 计算精确值。正的 `shots` 会对每个可观测量项采样，[`get_std`][fatqat.Result.get_std] 报告由此得到的标准误差。设置 `simulation_config["seed"]` 可以复现采样运行。

在后端上配置模拟方法、运行时和噪声。无效程序、可观测量宽度和 shot 值会在返回作业前引发 [`BackendValidationError`][fatqat.errors.BackendValidationError]；不受支持的可观测量类型会引发 `TypeError`。之后的故障由 [`result`][fatqat.Job.result] 抛出。

当程序会重置量子比特或应用通道噪声时，请使用密度矩阵后端。

使用 [`get_expectation`][fatqat.Result.get_expectation] 和 [`get_std`][fatqat.Result.get_std] 读取估计结果。如果还需要最终状态，请另行运行后端。

## 参数扫描


[`run_sweep`][fatqat.Estimator.run_sweep] 按输入顺序计算各绑定行。验证错误直接抛出；其他行故障由 [`result`][fatqat.Job.result] 抛出。不会返回部分结果列表。参数扫描引导请参阅[模拟量子程序](../guide/simulation.md)；可接受的绑定形状和随机种子行为在此处说明。

## 详细参考


::: fatqat.Observable
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.Estimator
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
