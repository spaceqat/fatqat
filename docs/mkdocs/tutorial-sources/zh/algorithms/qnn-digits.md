---
title: "使用量子神经网络识别手写数字"
description: "训练一个数据重上传电路来区分手写数字 3 和 6，同时通过一次扫描评估整批参数。"
icon: material-brain
figure_alts:
  - "平均池化后的手写数字输入"
  - "COBYLA 训练损失曲线"
  - "训练后对留出手写数字样本的预测"
---


# 使用量子神经网络识别手写数字


量子神经网络（QNN）分类器本质上是普通的参数化函数 $f(x;\theta)$，不同之处在于该函数由量子电路评估。输入特征 $x$ 和可训练权重 $\theta$ 都以旋转角的形式进入电路，各类别的得分则从最终态的期望值中读出。

本教程训练这样的分类器来区分手写数字 3 和 6，并展示 fatqat 相比“每个样本构建一条电路”的朴素工作流所提供的关键能力：**电路只构建一次，作为参数化模板；整批输入只需一次** `run_sweep` **调用即可评估**。

## 模型

在多轮操作中，每个量子比特交替施加*编码*门 $R_Y(x)$ 与*可训练*门 $R_Z(\theta)$，再用 CX 环将它们纠缠；这就是 Pérez-Salinas 等人提出的*数据重上传*模式（[arXiv:1907.02085](https://arxiv.org/abs/1907.02085)）。对于四个量子比特和四轮操作，

$$
|\psi(x;\theta)\rangle = U(x;\theta)\,|0\rangle^{\otimes 4},
\qquad
U(x;\theta) = \prod_{r=1}^{4} U_{\mathrm{ring}}
\left[\bigotimes_{q=0}^{3} R_Z(\theta_{r,q})\,R_Y(x_{r,q})\right],
$$

因此，16 个特征和 16 个权重每轮在每个量子比特上各使用一个。每轮都重新上传相同特征，使得小型电路也能拟合非线性决策边界。

读出方式遵循 Farhi 与 Neven 的方法（[arXiv:1802.06002](https://arxiv.org/abs/1802.06002)）：测量四个单量子比特期望值 $\langle Z_q\rangle$，并将其合并为两个类别的 logit：

$$
\ell_3 = \langle Z_0\rangle + \langle Z_1\rangle,
\qquad
\ell_6 = \langle Z_2\rangle + \langle Z_3\rangle,
\qquad
p = \mathrm{softmax}(\ell).
$$

训练通过无梯度的 COBYLA 优化器将平均交叉熵 $-\frac{1}{N}\sum_i \log p_i(\text{label}_i)$ 最小化。损失是一个黑盒，因此无需对电路求导。

数据是 scikit-learn 自带的 $8\times8$ 手写数字图像：它们是真实扫描结果，但已随软件包保存在本地，因此无需下载数据即可完整复现本页结果。所有随机源都设置了种子。

## 数据：两类小型数字图像

数字 3 和 6 组成的子集由带种子的生成器打乱，每张 $8\times8$ 图像通过平均池化缩小为 $4\times4$，每轮每个量子比特对应一个特征。像素值（0–16）被缩放到 $[0, \pi]$ 内的角度，因此空白像素编码为恒等旋转 $R_Y(0) = I$，而单个 $R_Y(x)$ 所产生的 $\langle Z\rangle = \cos x$ 中的 $\cos$ 则会保留亮度次序。

<!-- tutorial-code-cell -->

电路看到的是这个 $4\times4$ 池化结果，而非原始的 $8\times8$ 扫描图像。

<!-- tutorial-code-cell -->

## 试探态：一个参数化模板

[`ParameterVector`][fatqat.ParameterVector] 是一组具名占位符。添加量子门时以占位符代替数值，就会得到一个*模板*：其程序结构固定，角度留待之后绑定。绑定始终返回新程序，绝不改变模板，因此一个模板就能服务于所有样本和每个优化步骤。

<!-- tutorial-code-cell -->

该模板与其他程序一样可视化：每轮在每条量子线上先施加 `RY(x)`，再施加 `RZ(θ)`，并以 CX 环收尾，共重复四次。

<!-- tutorial-code-cell -->

## 用一次 `run_sweep` 评估整个批次

在一个优化步骤内，权重是固定的，因此只需用 `assign_parameters` 绑定一次。随后，批次扫描只改变特征：`run_sweep` 将完整的 `(N, 16)` 特征数组作为一次绑定传入，并返回有序的结果列表，每个样本对应一个普通 `Result`。参数已变成*数据*：普通 NumPy 数组流经一次调用，无需为每个样本重建电路结构。

（在当前 fatqat 版本中，`run_sweep` 仍然逐行降级并执行；目前的主要收益在于单模板工作流和一次调用的批处理接口，未来的融合批量执行也会接入这个接口。对以可观测量为中心的工作流，[`fatqat.Estimator`][fatqat.Estimator] 通过 `Estimator.run_sweep` 提供相同的批处理能力；请参阅[模拟指南](../guide/simulation.md)。）

<!-- tutorial-code-cell -->

先用随机初始权重做一个小检查：与未训练电路的预期一致，logit 从零附近起步。

<!-- tutorial-code-cell -->

## 训练

损失是训练批次上的平均 softmax 交叉熵。评估一次损失，就是对全部 120 张训练图像调用一次 `run_sweep`。

<!-- tutorial-code-cell -->

<!-- tutorial-code-cell -->

## 评估

在留出的测试集上评估训练后的权重；这里同样只需对整个批次扫描一次。

<!-- tutorial-code-cell -->

下面展示一组测试集预测样例，误分结果以红色标出。

<!-- tutorial-code-cell -->

## 下一步

* [Program 指南](../guide/program.md)介绍参数绑定；[模拟指南](../guide/simulation.md)讲解扫描，[解读结果](../guide/interpret-results.md)则涵盖精确与采样期望值。
* 很自然的扩展方向包括：增加轮数或量子比特，向模拟器传入噪声模型，或使用采样期望值（`shots > 0`）研究采样噪声对训练曲线的影响。
