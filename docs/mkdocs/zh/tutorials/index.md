---
title: "教程"
description: "从入门量子电路到多体动力学的可执行 fatqat 案例研究。"
---
<!-- 中文索引人工维护，与转换脚本生成的英文索引保持同步。 -->

# 教程

在面向具体任务的用户指南之外，这些完整且可复现的案例将带你进一步理解 fatqat。每个页面都交替展示原理说明和可执行的 Python 单元，并附有原始源文件，便于在本地继续探索。

!!! tip "选择学习路径"

    如果刚接触 fatqat，建议从贝尔态开始。算法篇会复用相同的参数与执行模型；中性原子篇则循序渐进，逐步贴近多体硬件的真实物理。

## 基础

从一个紧凑的量子电路入手：其精确态、采样结果与可视化解读都可以手工验证。

<div class="grid cards" markdown>

-   :material-set-split:{ .lg .middle } **制备并测量贝尔态**

    ---

    从精确振幅出发，得到固定随机种子下的测量计数，并将其与理想分布比较，完整追踪一个两比特贝尔态。

    [:material-arrow-right: 打开教程](bell-state.md)

</div>

## 算法

一次构建参数化程序，再通过优化器、扫描与估计器回答量子化学和机器学习问题。

<div class="grid cards" markdown>

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **使用 VQE 求解 H₂ 的基态能量**

    ---

    对氢分子运行精确、有限采样和含噪声的 VQE 循环，明确展示变分上界与采样不确定性。

    [:material-arrow-right: 打开教程](vqe-h2.md)

-   :material-brain:{ .lg .middle } **使用量子神经网络识别手写数字**

    ---

    训练一个数据重上传电路来区分手写数字 3 和 6，同时通过一次扫描评估整批参数。

    [:material-arrow-right: 打开教程](qnn-digits.md)

</div>

## 中性原子物理

从可编程连接逐步走向连续时间里德伯动力学与受约束的多体演化。

<div class="grid cards" markdown>

-   :material-atom:{ .lg .middle } **将八个原子纠缠为 GHZ 态**

    ---

    利用动态 `Pair` 和 `Unpair` 操作构建八原子 GHZ 态，然后同时检验其关联与相干相位。

    [:material-arrow-right: 打开教程](atom-array-ghz8.md)

-   :material-sine-wave:{ .lg .middle } **在里德伯原子链中建立反铁磁关联**

    ---

    从实际物理单位出发设计三阶段里德伯脉冲，观察短程反铁磁序如何在十个格点的原子链中出现。

    [:material-arrow-right: 打开教程](antiferromagnetic-chain.md)

-   :material-waveform:{ .lg .middle } **开放 PXP 链中的复苏与纠缠增长**

    ---

    对受约束的 PXP 哈密顿量进行 Trotter 分解，并将多体复苏与半链纠缠熵同独立的精确求解结果比较。

    [:material-arrow-right: 打开教程](pxp-z2-revival.md)

</div>
