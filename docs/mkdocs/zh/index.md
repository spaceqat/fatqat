---
template: home.html
hide:
  - navigation
  - toc
description: 只编写一个 FatQat Program，并为每项量子研究选择恰到好处的物理细节。
hero:
  eyebrow: 量子 SDK
  title: 一个 Program，三种物理细节层次
  summary: >-
    量子计算只需编写一次。随后无需更换编程模型，即可研究逻辑行为、测试设备约束，
    或追踪随时间演化的物理动力学。
  primary_action: 构建第一个 Program
  secondary_action: 比较执行层次
  install_action: 从源代码安装
  image_alt: 一个带测量与经典输出的双量子比特贝尔态线路。
  visual_note: 编写一次，运行时再选择所需细节。
  flow_label: 从 Program 到 Job 再到 Result
---

<!-- 本地化内容保留在 Markdown 中；共享的 Material 首屏位于 home.html。 -->

## 一套工作流，三种执行层次

每条路径都从同一个 [`Program`][fatqat.Program] 开始，并返回熟悉的 `Job` 与
`Result` 对象。只需选择足以回答当前问题的最低物理细节层次。

<div class="grid cards" markdown>

-   :material-chart-box-outline:{ .lg .middle } **通用模拟**

    ---

    当问题聚焦逻辑行为时，检查状态、计数与噪声。

    [:octicons-arrow-right-24: 了解通用模拟](guide/simulation.md)

-   :material-memory:{ .lg .middle } **硬件配置模拟**

    ---

    当设备约束很重要时，加入原生门与拓扑。

    [:octicons-arrow-right-24: 建模硬件配置](guide/hardware-profile-simulation.md)

-   :material-sine-wave:{ .lg .middle } **哈密顿量仿真**

    ---

    当物理行为很重要时，追踪脉冲、泄漏与动力学。

    [:octicons-arrow-right-24: 追踪物理动力学](guide/hamiltonian-emulation.md)

</div>

## 查看完整工作流

<div class="grid fatqat-home-example" markdown>

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()
```

<figure markdown>

![只包含相关的 00 和 11 贝尔态结果的柱状图。](assets/generated/guide/quickstart-counts.png)

<figcaption>固定随机种子的 1,000 次采样只返回相关结果。</figcaption>

</figure>

</div>

这段贝尔态 Program 展示了完整的线路级工作流。同一个编写对象还可以承载可复用
参数、混合局域维数、经典条件和直接物理控制。

## 探索文档

<div class="grid cards fatqat-home-destinations" markdown>

-   :material-play-circle-outline:{ .lg .middle } **快速上手**

    ---

    构建、绘制并运行第一个 Program。

    [:octicons-arrow-right-24: 开始构建](guide/quickstart.md)

-   :material-book-open-page-variant-outline:{ .lg .middle } **用户指南**

    ---

    学习核心概念和完整工作流。

    [:octicons-arrow-right-24: 阅读指南](guide/index.md)

-   :material-flask-outline:{ .lg .middle } **教程**

    ---

    探索可执行的算法与物理案例。

    [:octicons-arrow-right-24: 运行教程](tutorials/index.md)

-   :material-format-list-bulleted:{ .lg .middle } **API 参考**

    ---

    查找准确的签名与校验约定。

    [:octicons-arrow-right-24: 查阅 API](api/index.md)

</div>
