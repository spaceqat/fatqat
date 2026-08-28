# 模拟量子程序

使用通用 [`Simulator`][fatqat.simulator.Simulator] 研究逻辑线路演化，而无需
选择硬件配置或哈密顿量。它会按原样应用 [`Program`][fatqat.Program] 中的
操作：不会进行转译、路由，也不会附加设备时序。

一个可复用的旋转会带我们依次完成状态计算、测量采样和参数扫描。

## 从可复用的 Program 开始

描述计算时，先让角度保持为符号。绑定会创建一个新的 Program，因此模板仍可
用于其他值：

```pycon
>>> import math
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> theta = fq.Parameter("theta")
>>> rotation = fq.Program(1, 1)
>>> rotation.add(ops.RY(theta), 0)
>>> bound = rotation.assign_parameters({theta: math.pi / 2})
>>> backend = fq.simulator.Simulator(method="statevector", runtime="numpy")
>>> result = backend.run(bound).result()
>>> np.round(np.abs(result.get_statevector()) ** 2, 6).tolist()
[0.5, 0.5]
```

`RY(pi / 2)` 制备出 `|0>` 与 `|1>` 概率相等的状态。Program 尚未被测量，
因此最自然的答案是其最终状态。`runtime="numpy"` 让这个小例子免于编译启动
开销；[性能与扩展](performance.md)会说明何时应将它与 Numba 运行时比较。

请按所需输出选择表示方法：

| 需要了解的问题 | 建议从这里开始 |
| --- | --- |
| 理想线路制备的纯态 | `method="statevector"` |
| 有限噪声通道作用后的精确混态 | `method="density_matrix"` |
| 小型 Program 实现的相干变换 | `method="unitary"` |
| 小型 Program 实现的完整通道 | `method="superop"` |

这些是逻辑演化的不同视图，并非编写计算的不同方式。

## 测量一个分布

若要查看实验会报告的结果，请复制已绑定的 Program，追加测量并请求重复采样：

```pycon
>>> measured = bound.copy()
>>> measured.measure(0, 0)
>>> counts = backend.run(
...     measured,
...     shots=200,
...     simulation_config={"seed": 7},
... ).result().get_counts()
>>> sum(counts.values())
200
>>> set(counts) <= {"0", "1"}
True
```

两种结果的频率会在相等附近波动。随机种子让这次运行可复现，但代码通常应检验
物理性质——允许出现的结果和总采样数——而不是某一个精确的随机字典。关于
计数顺序以及如何在 Result 中存储的多种答案间选择，请参阅[解读一次运行](interpret-results.md)。

## 无需重建即可扫描

[`run_sweep`][fatqat.simulator.Simulator.run_sweep] 会将每行数值绑定到同一个
Program 结构。此处状态本身就是有用的答案，因此无需测量或采样：

```pycon
>>> angles = np.linspace(0.0, 2.0 * np.pi, 9)
>>> sweep = backend.run_sweep(
...     rotation,
...     {theta: angles},
...     result_config={"counts": False, "final_state": True},
... ).result()
>>> probability_one = np.array([
...     abs(item.get_statevector()[1]) ** 2 for item in sweep
... ])
>>> np.round(probability_one[[0, 4, 8]], 6).tolist()
[0.0, 1.0, 0.0]
```

完整响应曲线清楚展示了这种复用：

![当 RY 角度从零扫描到二倍 pi 时，测得一的概率沿平滑的正弦平方曲线变化。](../assets/generated/guide/simulation-1.png)

??? example "复现此图"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    theta = fq.Parameter("theta")
    rotation = fq.Program(1, 1)
    rotation.add(ops.RY(theta), 0)

    angles = np.linspace(0.0, 2.0 * np.pi, 41)
    backend = fq.simulator.Simulator(method="statevector", runtime="numpy")
    results = backend.run_sweep(
        rotation,
        {theta: angles},
        result_config={"counts": False, "final_state": True},
    ).result()
    probability_one = np.array([
        abs(result.get_statevector()[1]) ** 2 for result in results
    ])

    assert np.allclose(probability_one, np.sin(angles / 2.0) ** 2)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(angles, probability_one, color="#3b6ea8", linewidth=2)
    ax.set(
        xlabel=r"rotation angle $\theta$",
        ylabel=r"$P(1)$",
        xlim=(0.0, 2.0 * np.pi),
        ylim=(-0.03, 1.03),
    )
    ax.set_xticks(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    ```

扫描会按输入顺序返回普通 Results。接受的方法与批处理形式请参阅
[Simulator API](../api/simulator.md)。接下来，[解读一次运行](interpret-results.md)
会沿着共同的 Job 和 Result 边界，依次说明计数、状态、映射和期望值。
