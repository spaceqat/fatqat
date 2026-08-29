---
title: "开放 PXP 链中的复苏与纠缠增长"
description: "对受约束的 PXP 哈密顿量进行 Trotter 分解，并将多体复苏和半链纠缠熵与独立精确求解结果比较。"
icon: material-waveform
figure_alts:
  - "PXP 复苏保真度、纠缠熵与位点占据"
---


# 开放 PXP 链中的复苏与纠缠增长


将里德伯阻塞推到强耦合极限，所得到的就是 PXP 模型。从 Néel 态对它进行量子淬火，会展现出多体物理中最奇异的景象之一。

直觉上，十格点链会迅速扰乱并忘记初始态。PXP 链在大多数情况下确实如此——但从 $|Z_2\rangle = |r\,g\,r\,g\,\ldots\rangle$ 出发时是例外。该状态会以规律间隔突然回到自身，这一现象与量子多体疤痕有关（参见 [Turner 等，Nature Physics 14, 745 (2018)](https://doi.org/10.1038/s41567-018-0137-5)）。淬火过程中纠缠会增长，但并非只增不减：每次复苏都伴随半链纠缠熵的明显下降。

实际操作中有一个限制：fatqat 的脉冲仿真器始终从全基态出发，且只提供全局控制，因此无法承载从 $|Z_2\rangle$ 开始的淬火。下面的解决方案是将 PXP 哈密顿量 Trotter 分解为小型自定义门，再在门级模拟器上运行；该模拟器可以接受任意 `initial_state`。同时，对精确 PXP 模型作独立 QuTiP 求解，沿途检验每条曲线。

## 1. PXP 的来源：里德伯哈密顿量的阻塞极限

从里德伯哈密顿量出发：

$$
H(t) = \frac{\Omega(t)}{2}\sum_i X_i
       - \Delta(t)\sum_i n_i
       + \sum_{i<j} U_{ij} n_i n_j,
\qquad
n_i=|r\rangle\langle r|_i.
$$

现在将 $U$ 调大并设置 $\Delta=0$。当两个相邻激发的能量代价超过问题中其他所有尺度后，“两个相邻格点不能同时处于 $|r\rangle$”就不再只是偏好，而成为硬约束。在该受约束子空间内，一阶微扰理论只留下

$$
H_{\mathrm{PXP}} = \frac{\Omega}{2}\sum_i P_{i-1} X_i P_{i+1},
\qquad
P_i = |g\rangle\langle g|_i = I - n_i,
$$

并采用开放边界，也就是 $P_{-1}=P_L=1$。直观解读每一项：格点 $i$ 可以翻转，但仅当其两个邻居都处于 $|g\rangle$ 时才能翻转。体内项涉及三个格点；两个边界项 $X_0 P_1$ 和 $P_{L-2}X_{L-1}$ 只涉及两个。

需要明确本例的范围：这是理想模型研究。约束是精确施加的（而非用很大但有限的 $U$ 近似），且不包含退相干和原子丢失。

## 2. Z2 态及其复苏原因

受约束子空间中存在两种 Néel 构型：

$$
|Z_2\rangle = |r\,g\,r\,g\,\ldots\rangle,
\qquad
|\bar Z_2\rangle = |g\,r\,g\,r\,\ldots\rangle.
$$

我们制备 $|Z_2\rangle$，使其自由演化，并观察返回概率 $F(t)=|\langle Z_2|\psi(t)\rangle|^2$。少数特殊的疤痕本征态主导了这次淬火，所以 $F(t)$ 以接近 $T\approx 4.7/g$ 的周期振荡，其中 $g=\Omega/2$ 是 PXP 系数。对下文使用的驱动，这意味着 $T\approx 1.5$ `us`。不过，我们不会盲信估算，而是直接测量峰值实际出现的位置。

## 导入、常量与 Z2 向量

本例包含十个格点，使用 `rad/us` 和 `us` 单位，并采用熟悉的驱动尺度 $\Omega=2\pi$ `rad/us`，因此 PXP 系数是 $g=\pi$ `rad/us`。需要牢记一个约定：fatqat 将 $|b_0\ldots b_9\rangle$ 的振幅存放在索引 $\sum_i b_i 2^i$ 处（格点 0 是最低有效位）。因此，每个 Néel 态都是一个单独基向量，只需记住两个索引值。

<!-- tutorial-code-cell -->

## 3. 将 PXP 进行 Trotter 分解并写成 fatqat 程序

两个限制决定了实现方式。首先，脉冲仿真器始终从 $|g\ldots g\rangle$ 开始，并且只提供全局控制，所以根本无法在其中表示从 $|Z_2\rangle$ 出发的淬火；相比之下，门级 [`Simulator`][fatqat.simulator.Simulator] 可以直接接受 `initial_state`。其次，任何原生门集都不包含 PXP 项，但这正是 fatqat 自定义操作扩展点的用途：[`MatrixImplementationMap`][fatqat.implementation.MatrixImplementationMap] 在执行时将任意固定元数的操作族解析为局部矩阵。

所需的指数算符很容易写出。令 $M=PXP$ 且 $M^2=P\otimes I\otimes P$，则

$$
e^{-i\theta M}
  = I + (\cos\theta - 1)\,P\otimes I\otimes P - i\sin\theta\,M.
$$

具体而言，对体内项，这是 $|ggg\rangle$ 与 $|grg\rangle$ 之间的 $2\times 2$ 旋转；对边界项，则是 $|gg\rangle$ 与 $|rg\rangle$（左边缘）或 $|gg\rangle$ 与 $|gr\rangle$（右边缘）之间的旋转。每个矩阵只构建一次，通过幺正性检查后注册到对应操作类下。

现在考察真实时间演化。我们需要 $e^{-iH_{\mathrm{PXP}}dt}$，其中 $H_{\mathrm{PXP}}=\sum_j h_j$，$h_j=(\Omega/2)M_j$。不同 $h_j$ 之间不对易，因此指数算符无法分解为各个 $e^{-ih_j dt}$ 的精确乘积，必须进行近似。一阶 Trotter 公式只对各项扫描一次：

$$
e^{-iH dt} \;\approx\;
\prod_{j=0}^{L-1} e^{-i h_j dt}
\;=\; e^{-iH dt} + \mathcal{O}(dt^2),
$$

其领头误差是 $h_j$ 之间的对易子。对称（Strang）步骤可以消除该领头误差：将每项拆成两个半角，先正向扫描，再反向扫描相同的半角：

$$
e^{-iH dt} \;\approx\;
\left(\prod_{j=0}^{L-1} e^{-i h_j dt/2}\right)
\left(\prod_{j=L-1}^{0} e^{-i h_j dt/2}\right)
\;=\; e^{-iH dt} + \mathcal{O}(dt^3).
$$

在代码中，正向遍历是 `for site in range(NUM_SITES)`，依次处理左边缘、体内项、右边缘；反向遍历则倒序执行同一循环。每个半步的指数是 $h_j\,dt/2$，而 $h_j$ 已包含 $\Omega/2$，因此每个已注册矩阵都使用同一个角 $\theta=\Omega\,dt/4$。将这个对称步骤重复 `round(duration / dt)` 次，就能将淬火推进到任意持续时间。

根据实践经验，这里有一点需要警惕：fatqat 将局部矩阵展平时，*第一个*目标是最高有效位，索引为 $b_{t_0}\cdot 2^{k-1} + \cdots + b_{t_{k-1}}$。这个顺序很容易颠倒（我们就曾做错），并且结果是无声地得到错误动力学，而不是收到错误信息。

<!-- tutorial-code-cell -->

## 4. 精确参考求解

手工构建的 Trotter 电路在经过独立检验前绝不应被盲目信任。fatqat 自身的物理测试会将数值结果与独立参考求解对照，因此我们也采用相同做法：直接用 QuTiP 组装精确 PXP 哈密顿量，从同一个 $|Z_2\rangle$ 向量出发，在完整时间网格上一次求解。

<!-- tutorial-code-cell -->

## 5. 测量对象：保真度与半链纠缠熵

两个数值就能讲清完整故事。保真度 $F(t)=|\langle Z_2|\psi(t)\rangle|^2$ 表示有多少状态分量回到初始态，而与双生分支的保真度则告诉我们，复苏落在*相同*的 Néel 取向上，还是落在镜像取向上。半链 von Neumann 熵

$$
S(t) = -\mathrm{Tr}\left[\rho_A\ln\rho_A\right],
\qquad
\rho_A = \mathrm{Tr}_B|\psi(t)\rangle\langle\psi(t)|,
$$

中 $A$ 表示前五个格点，该量用于追踪链两半之间的纠缠程度。两条曲线都由同样的小型辅助函数得到，因此可以直接比较 fatqat 结果与参考求解数值。

<!-- tutorial-code-cell -->

## 6. 运行淬火并收集时间序列

对每个时间网格点，我们重建 Trotter 程序（其长度随持续时间增长），并从 $|Z_2\rangle$ 出发演化一次；参考求解则给出对应的精确态。这种时间轴采样方式有些直接粗暴，但能使全部过程都保持在公开 API 之上，而且十格点链足够小，数秒就能运行完毕。

<!-- tutorial-code-cell -->

复苏峰就是初始衰减后保真度的局部极大值。请注意，每个峰值都落在*相同*的 Néel 分支上；此时双生分支的保真度始终可忽略，所以该状态确实回到了出发点。

<!-- tutorial-code-cell -->

## 7. 解读复苏与纠缠熵增长

左图是最关键的结果：在 $t\approx 1.5, 3.0, 4.5$ `us` 附近出现三次复苏，强度一次比一次稍弱，这是 PXP 疤痕的指纹。中图说明这些动力学为何不属于热化：半链熵虽然增长，却会在每次复苏时*下降*，就像该状态短暂地重新想起了如何成为直积态。右图在实空间中讲述同一个故事：Néel 条纹在淬火过程中融化，又在每次保真度达到峰值时部分重组。

<!-- tutorial-code-cell -->
