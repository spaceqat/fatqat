---
hide:
  - navigation
  - toc
---

<!-- Material 专用首页；产品叙事应与 docs/sphinx/index.md 保持一致。 -->

<div class="fatqat-home" markdown>

<section class="fatqat-home__hero" aria-labelledby="fatqat-home-title" markdown>

<div class="fatqat-home__hero-copy" markdown>

<p class="fatqat-home__eyebrow">量子 SDK</p>

# 一个 Program，三种物理细节层次 { #fatqat-home-title }

<p class="fatqat-home__summary" markdown>
只需将量子计算编写一次，保存为一个 [`Program`][fatqat.Program]。随后无需更换
编程模型，即可研究逻辑行为、测试设备约束，或追踪随时间演化的物理动力学。
</p>

<div class="fatqat-home__actions" markdown>
[构建第一个 Program](guide/quickstart.md){ .md-button .md-button--primary }
[比较执行层次](guide/execution-models.md){ .md-button }
</div>

<p class="fatqat-home__requirements" markdown>
Python 3.12+ · [从源代码安装](guide/quickstart.md)
</p>

</div>

<div class="fatqat-home__model" role="img" aria-label="同一个 FatQat Program 可选择通用模拟、硬件配置模拟或哈密顿量仿真三种执行层次，且每条路径都返回 Job 与 Result。">
<div class="fatqat-home__model-node"><code>Program</code></div>
<div class="fatqat-home__model-connector" aria-hidden="true"></div>
<div class="fatqat-home__model-choice">选择一种执行层次</div>
<div class="fatqat-home__model-targets">
<div class="fatqat-home__model-target">
<strong>通用模拟</strong>
<span>状态 · 计数 · 噪声</span>
</div>
<div class="fatqat-home__model-target">
<strong>硬件配置模拟</strong>
<span>原生门 · 拓扑</span>
</div>
<div class="fatqat-home__model-target">
<strong>哈密顿量仿真</strong>
<span>脉冲 · 泄漏 · 动力学</span>
</div>
</div>
<div class="fatqat-home__model-connector" aria-hidden="true"></div>
<div class="fatqat-home__model-node"><code>Job</code> → <code>Result</code></div>
</div>

</section>

<div class="grid cards fatqat-home__benefits" markdown>

-   :material-vector-combine: **统一编写对象**

    将门、测量、条件、参数、量子多能级系统和物理控制都保存在同一个 Program 中。

-   :material-tune-variant: **按需选择精度**

    只建模当前问题真正需要的物理细节。

-   :material-swap-horizontal-bold: **工作流始终一致**

    始终通过相同的 `Job` 与 `Result` 概念提交任务、检查输出。

</div>

## 查看完整工作流

<div class="grid fatqat-home__workflow" markdown>

<div markdown>

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

</div>

<div class="fatqat-home__workflow-result" markdown>

![只包含相关的 00 和 11 贝尔态结果的柱状图。](assets/generated/guide/quickstart-counts.png)

<p class="fatqat-home__workflow-caption">固定随机种子的 1,000 次采样只返回相关结果。</p>

</div>

</div>

这段贝尔态 Program 展示了完整的线路级工作流。同一个 Program 抽象还可以承载
可复用参数、混合局域维数、经典条件和直接物理控制。

## 选择下一步

<div class="grid cards fatqat-home__destinations" markdown>

-   :material-play-circle-outline: **[快速上手](guide/quickstart.md)**

    构建、绘制并运行第一个 Program。

-   :material-book-open-page-variant-outline: **[用户指南](guide/index.md)**

    学习核心概念和完整工作流。

-   :material-flask-outline: **[教程](tutorials/index.md)**

    探索可执行的算法与物理案例。

-   :material-format-list-bulleted: **[API 参考](api/index.md)**

    查找准确的签名与校验约定。

</div>

</div>
