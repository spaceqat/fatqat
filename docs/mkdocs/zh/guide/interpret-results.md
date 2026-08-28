# 解读一次运行

每次 FatQat 执行都会跨越同一个边界：提交 [`Program`][fatqat.Program]，收到
[`Job`][fatqat.Job]，再调用 [`result`][fatqat.Job.result] 获取答案。哪种结果
有用，取决于你提出的问题。

我们会构建一个贝尔 Program，并用四种方式使用它：观测结果、最终状态、实现的
映射和期望值。

## 只构建一次计算

开始时先不要测量 Program。这样，关于状态、映射和可观测量的问题都仍然开放：

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> bell = fq.Program(2, 2)
>>> bell.add(ops.H, 0)
>>> bell.add(ops.CX, (0, 1))
>>> backend = fq.simulator.Simulator(method="statevector", runtime="numpy")
```

## 出现了哪些结果？

如果想要的输出是采样得到的经典分布，请复制 Program 并追加测量：

```pycon
>>> measured_bell = bell.copy()
>>> measured_bell.measure_all()
>>> counts_result = backend.run(
...     measured_bell,
...     shots=400,
...     simulation_config={"seed": 7},
... ).result()
>>> counts = counts_result.get_counts()
>>> sum(counts.values())
400
>>> set(counts) <= {"00", "11"}
True
```

只会出现相互关联的贝尔结果。计数字符串把编号最高的经典槽位显示在左侧，把
槽位 0 显示在右侧。一个刻意设置得不对称的示例可以清楚展示这种顺序：

```pycon
>>> order_demo = fq.Program(2, 2)
>>> order_demo.add(ops.X, 0)
>>> order_demo.measure_all()
>>> backend.run(
...     order_demo,
...     shots=8,
...     simulation_config={"seed": 7},
... ).result().get_counts()
{'01': 8}
```

量子比特 0 为 `1`，因此经典槽位 0 出现在最右一位。如果代码更适合让槽位 0
排在最前，而不是使用显示字符串，请调用
[`get_counts_as_tuples`][fatqat.Result.get_counts_as_tuples]。

## 运行最终到达了什么状态？

运行未测量的 Program，并请求其自然的最终状态：

```pycon
>>> state_result = backend.run(
...     bell,
...     result_config={"counts": False, "final_state": True},
... ).result()
>>> sorted(state_result.available_data)
['statevector']
>>> state = state_result.get_statevector()
>>> np.round(np.abs(state) ** 2, 6).tolist()
[0.5, 0.0, 0.0, 0.5]
```

在选择访问器之前，`available_data` 会告诉你此次运行实际生成了哪些数据。这里，
状态在 `|00>` 和 `|11>` 上的概率相等。密度矩阵运行会以矩阵表示同一个纯态，
也能表示精确的混态演化。哈密顿量仿真器返回的状态可能包含非计算物理能级，
或包含逻辑 Program 未寻址的模型子系统。

## Program 实现了什么变换？

未测量的相干 Program 本身也可以成为研究对象。以酉矩阵方法运行后，返回矩阵
的第一列就是 Program 从 `|00>` 制备的状态：

```pycon
>>> unitary = (
...     fq.simulator.Simulator(method="unitary", runtime="numpy")
...     .run(bell)
...     .result()
...     .get_unitary()
... )
>>> np.allclose(unitary[:, 0], state)
True
```

当小型相干模块的完整作用很重要时，请使用这条路径。超算符把这一思路扩展到
完整通道，但计算成本会显著增加。

## 我关心的是哪一个物理量？

如果答案本来就是相关性或磁化强度等期望值，计数只是一种间接表示。
[`Estimator`][fatqat.Estimator] 通过后端演化未测量的 Program，并在结果状态
上计算 [`Observable`][fatqat.Observable]：

```pycon
>>> estimator = fq.Estimator(
...     fq.simulator.Simulator(method="statevector", runtime="numpy")
... )
>>> zz = fq.Observable([("ZZ", 1.0)])
>>> z_on_qubit_zero = fq.Observable([("IZ", 1.0)])
>>> exact = estimator.run(bell, [zz, z_on_qubit_zero]).result()
>>> np.round(exact.get_expectation(), 6).tolist()
[1.0, 0.0]
```

贝尔对完全关联，因此 `<ZZ> = 1`。每个单独量子比特在零和一之间均衡，因此
`<Z_0> = 0`。可观测量标签把量子比特 0 放在右侧，与计数字符串顺序一致。

默认情况下，Estimator 根据最终状态计算精确值。改用正的采样次数，则可看到
有限采样请求的统计精度：

```pycon
>>> sampled = estimator.run(
...     bell,
...     z_on_qubit_zero,
...     shots=400,
...     simulation_config={"seed": 7},
... ).result()
>>> bool(abs(sampled.get_expectation()) < 0.2)
True
>>> sampled.get_std() > 0.0
True
```

估计值会在零附近波动，`get_std()` 报告其标准误差。当限制因素是统计精度而非
状态演化时，请增加采样次数。

关于正式的状态轴、算符向量化和可观测量形状契约，请查阅 [Result](../api/result.md)、
[Simulator](../api/simulator.md) 和 [Estimator](../api/estimator.md) 参考。

下一个常见问题是，这一答案能否经受实际误差。[在理想与含噪条件下比较同一
Program](ideal-and-noisy.md)会改变执行模型，同时保持贝尔 Program 不变。
