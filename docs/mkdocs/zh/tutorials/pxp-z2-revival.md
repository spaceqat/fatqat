---
title: "开放 PXP 链中的复苏与纠缠增长"
description: "对受约束的 PXP 哈密顿量进行 Trotter 分解，并将多体复苏和半链纠缠熵与独立精确求解结果比较。"
---
<!-- 中文译文人工维护；运行结果由 docs/mkdocs/tools/convert_tutorials.py 从规范源码同步。 -->

# 开放 PXP 链中的复苏与纠缠增长

<div class="grid cards" markdown>

-   :material-map-marker-path: **学习路径**

    中性原子物理

-   :material-language-python: **可执行源码**

    [下载 `plot_pxp_z2_revival.py`](../downloads/tutorials/plot_pxp_z2_revival.py){ download }

</div>

将里德伯阻塞推到强耦合极限，所得到的就是 PXP 模型。从 Néel 态对它进行量子淬火，会展现出多体物理中最奇异的景象之一。

直觉上，十格点链会迅速扰乱并忘记初始态。PXP 链在大多数情况下确实如此——但从 $|Z_2\rangle = |r\,g\,r\,g\,\ldots\rangle$ 出发时是例外。该状态会以规律间隔突然回到自身，这一现象与量子多体疤痕有关（参见 [Turner 等，Nature Physics 14, 745 (2018)](https://doi.org/10.1038/s41567-018-0137-5)）。淬火过程中纠缠会增长，但并非只增不减：每次复苏都伴随半链纠缠熵的明显下降。

实际操作中有一个限制：fatqat 的脉冲仿真器始终从全基态出发，且只提供全局控制，因此无法承载从 $|Z_2\rangle$ 开始的淬火。下面的解决方案是将 PXP 哈密顿量 Trotter 分解为小型自定义门，再在门级模拟器上运行；该模拟器可以接受任意 `initial_state`。同时，对精确 PXP 模型作独立 QuTiP 求解，沿途检验每条曲线。

!!! info "基于源码的教程"

    说明文字是对版本库中教程源码的人工中文翻译，页面中的可执行单元保留规范源码。转换脚本从同一源码捕获运行结果；其中的英文标签来自源码的打印语句，保留原样以便核对。页面不显示仅用于文档验证的代码段。下载并直接运行 Python 文件即可复现图形与标准输出。

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

```python title="Python 单元 1"
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

import fatqat as fq

NUM_SITES = 10
HALF = NUM_SITES // 2
DIM = 2**NUM_SITES

OMEGA = 2 * np.pi  # rad/us -> PXP coefficient g = OMEGA / 2
T_MAX = 6.0  # us, covers the first three revivals
DT_TROTTER = 0.01  # us, one symmetric second-order Trotter step
TIME_GRID = np.linspace(0.0, T_MAX, 121)

Z2_BITS = tuple(1 - (i % 2) for i in range(NUM_SITES))  # |r g r g ...>
Z2_INDEX = sum(2**i for i in range(NUM_SITES) if Z2_BITS[i])
Z2 = np.zeros(DIM, dtype=complex)
Z2[Z2_INDEX] = 1.0

ALT_BITS = tuple(1 - bit for bit in Z2_BITS)  # the twin Neel branch
ALT_INDEX = sum(2**i for i in range(NUM_SITES) if ALT_BITS[i])
ALT = np.zeros(DIM, dtype=complex)
ALT[ALT_INDEX] = 1.0

print(f"Z2 branch |r g r g ...> sits at statevector index {Z2_INDEX}")
print(f"twin branch |g r g r ...> sits at statevector index {ALT_INDEX}")
```

<!-- tutorial-result-start:cell-1 -->
!!! example "运行结果"

    ```text
    Z2 branch |r g r g ...> sits at statevector index 341
    twin branch |g r g r ...> sits at statevector index 682
    ```

<!-- tutorial-result-end:cell-1 -->

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

```python title="Python 单元 2"
from dataclasses import dataclass
from typing import ClassVar

from fatqat.operations import Operation

THETA = OMEGA * DT_TROTTER / 4


@dataclass(frozen=True)
class PXPBulk(Operation):
    """Three-site PXP exponential: X on the middle site guarded by two P's."""

    name: ClassVar[str] = "PXPBulk"
    num_subsystems: ClassVar[int] = 3


@dataclass(frozen=True)
class PXPEdgeLeft(Operation):
    """Left boundary term X_0 P_1."""

    name: ClassVar[str] = "PXPEdgeLeft"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class PXPEdgeRight(Operation):
    """Right boundary term P_{L-2} X_{L-1}."""

    name: ClassVar[str] = "PXPEdgeRight"
    num_subsystems: ClassVar[int] = 2


def _rotation_matrix(dimension: int, pair: tuple[int, int], angle: float) -> np.ndarray:
    """Identity except for a 2x2 exp(-i angle X) rotation on ``pair``."""
    matrix = np.eye(dimension, dtype=complex)
    first, second = pair
    matrix[first, first] = matrix[second, second] = np.cos(angle)
    matrix[first, second] = matrix[second, first] = -1j * np.sin(angle)
    return matrix


# fatqat flattens local matrices with the FIRST target as the most
# significant bit: index = b_{t0} * 2**(k-1) + ... + b_{t_{k-1}}.
# Keep that in mind or the edge terms quietly act on the wrong site.
# Bulk (i-1, i, i+1): flip the middle site -> pair (|ggg>, |grg>) = (0, 2).
BULK_MATRIX = _rotation_matrix(8, (0, 2), THETA)
# Left edge (0, 1): flip site 0 -> pair (|gg>, |rg>) = (0, 2).
EDGE_LEFT_MATRIX = _rotation_matrix(4, (0, 2), THETA)
# Right edge (L-2, L-1): flip site L-1 -> pair (|gg>, |gr>) = (0, 1).
EDGE_RIGHT_MATRIX = _rotation_matrix(4, (0, 1), THETA)


implementation_map = fq.implementation.MatrixImplementationMap()
implementation_map.add(PXPBulk, BULK_MATRIX)
implementation_map.add(PXPEdgeLeft, EDGE_LEFT_MATRIX)
implementation_map.add(PXPEdgeRight, EDGE_RIGHT_MATRIX)

# NumPy runtime on purpose: the program is thousands of tiny local
# operations, and at that size the Python loop dwarfs any matrix kernel,
# so numba would only add compile time.
backend = fq.simulator.Simulator(
    method="SV",
    runtime="numpy",
    implementation_map=implementation_map,
)


def trotter_program(duration: float) -> fq.Program:
    """Build the symmetric second-order Trotter program for ``duration``."""
    steps = int(round(duration / DT_TROTTER))
    program = fq.Program(NUM_SITES)
    for _ in range(steps):
        for site in range(NUM_SITES):  # forward half pass
            if site == 0:
                program.add(PXPEdgeLeft(), (0, 1))
            elif site == NUM_SITES - 1:
                program.add(PXPEdgeRight(), (NUM_SITES - 2, NUM_SITES - 1))
            else:
                program.add(PXPBulk(), (site - 1, site, site + 1))
        for site in range(NUM_SITES - 1, -1, -1):  # backward half pass
            if site == 0:
                program.add(PXPEdgeLeft(), (0, 1))
            elif site == NUM_SITES - 1:
                program.add(PXPEdgeRight(), (NUM_SITES - 2, NUM_SITES - 1))
            else:
                program.add(PXPBulk(), (site - 1, site, site + 1))
    return program


def evolve_fatqat(duration: float) -> np.ndarray:
    """Quench |Z2> for ``duration`` and return the final statevector."""
    if duration == 0.0:
        return Z2.copy()
    result = backend.run(
        trotter_program(duration),
        initial_state=Z2,
        result_config={"counts": False, "final_state": True},
    ).result()
    return result.get_statevector()
```

<!-- tutorial-result-start:cell-2 -->
!!! example "运行结果"

    ```text
    bulk term matrix is unitary (8x8)
    left edge term matrix is unitary (4x4)
    right edge term matrix is unitary (4x4)
    ```

<!-- tutorial-result-end:cell-2 -->

## 4. 精确参考求解

手工构建的 Trotter 电路在经过独立检验前绝不应被盲目信任。fatqat 自身的物理测试会将数值结果与独立参考求解对照，因此我们也采用相同做法：直接用 QuTiP 组装精确 PXP 哈密顿量，从同一个 $|Z_2\rangle$ 向量出发，在完整时间网格上一次求解。

```python title="Python 单元 3"
import qutip

_P = (qutip.qeye(2) + qutip.sigmaz()) / 2  # |g><g|
_X = qutip.sigmax()
_I2 = qutip.qeye(2)


def _pxp_term(site: int) -> "qutip.Qobj":
    factors = [_I2] * NUM_SITES
    factors[site] = _X
    if site - 1 >= 0:
        factors[site - 1] = _P
    if site + 1 < NUM_SITES:
        factors[site + 1] = _P
    return qutip.tensor(*factors)


ORACLE_H = sum((OMEGA / 2) * _pxp_term(site) for site in range(NUM_SITES))
ORACLE_Z2 = qutip.tensor(*[qutip.basis(2, bit) for bit in Z2_BITS])
ORACLE_RESULT = qutip.sesolve(ORACLE_H, ORACLE_Z2, list(TIME_GRID))

# Gotcha: QuTiP's tensor() treats the FIRST factor as the most significant
# bit, while fatqat puts site 0 at the least significant bit. Same physics,
# differently labelled basis -- so every oracle vector is bit-reversed into
# fatqat's convention before we compare anything.
def _bit_reversal_permutation() -> np.ndarray:
    permutation = np.empty(DIM, dtype=int)
    for index in range(DIM):
        reversed_index = 0
        for site in range(NUM_SITES):
            reversed_index |= ((index >> site) & 1) << (NUM_SITES - 1 - site)
        permutation[index] = reversed_index
    return permutation


_ORACLE_PERMUTATION = _bit_reversal_permutation()


def evolve_oracle(index: int) -> np.ndarray:
    """Return the oracle statevector at grid index ``index``, fatqat-ordered."""
    vector = np.asarray(ORACLE_RESULT.states[index].full()).reshape(-1)
    return vector[_ORACLE_PERMUTATION]
```

## 5. 测量对象：保真度与半链纠缠熵

两个数值就能讲清完整故事。保真度 $F(t)=|\langle Z_2|\psi(t)\rangle|^2$ 表示有多少状态分量回到初始态，而与双生分支的保真度则告诉我们，复苏落在*相同*的 Néel 取向上，还是落在镜像取向上。半链 von Neumann 熵

$$
S(t) = -\mathrm{Tr}\left[\rho_A\ln\rho_A\right],
\qquad
\rho_A = \mathrm{Tr}_B|\psi(t)\rangle\langle\psi(t)|,
$$

中 $A$ 表示前五个格点，该量用于追踪链两半之间的纠缠程度。两条曲线都由同样的小型辅助函数得到，因此可以直接比较 fatqat 结果与参考求解数值。

```python title="Python 单元 4"
def fidelity(state: np.ndarray, reference: np.ndarray) -> float:
    """Return |<reference|state>|^2 for two statevectors."""
    return float(abs(np.vdot(reference, state)) ** 2)


def half_chain_entropy(state: np.ndarray) -> float:
    """Von Neumann entropy of the first-half subsystem, via the SVD."""
    schmidt = np.linalg.svd(state.reshape(2**HALF, 2**HALF), compute_uv=False)
    probabilities = schmidt**2
    probabilities = probabilities[probabilities > 0.0]
    return -float(np.sum(probabilities * np.log(probabilities)))


def site_occupations(state: np.ndarray) -> np.ndarray:
    """Marginal |r> population of every site from one statevector."""
    basis_indices = np.arange(DIM)
    return np.array(
        [
            float(np.sum(np.abs(state) ** 2 * ((basis_indices >> site) & 1)))
            for site in range(NUM_SITES)
        ]
    )
```

## 6. 运行淬火并收集时间序列

对每个时间网格点，我们重建 Trotter 程序（其长度随持续时间增长），并从 $|Z_2\rangle$ 出发演化一次；参考求解则给出对应的精确态。这种时间轴采样方式有些直接粗暴，但能使全部过程都保持在公开 API 之上，而且十格点链足够小，数秒就能运行完毕。

```python title="Python 单元 5"
fatqat_states = [evolve_fatqat(t) for t in TIME_GRID]

fatqat_fidelity = np.array([fidelity(s, Z2) for s in fatqat_states])
fatqat_alt = np.array([fidelity(s, ALT) for s in fatqat_states])
fatqat_entropy = np.array([half_chain_entropy(s) for s in fatqat_states])
fatqat_occupations = np.array([site_occupations(s) for s in fatqat_states])

oracle_fidelity = np.array(
    [fidelity(evolve_oracle(i), Z2) for i in range(len(TIME_GRID))]
)
oracle_entropy = np.array(
    [half_chain_entropy(evolve_oracle(i)) for i in range(len(TIME_GRID))]
)

max_gap = float(np.max(np.abs(fatqat_fidelity - oracle_fidelity)))
print(f"Largest fidelity gap between Trotter and oracle: {max_gap:.5f}")
```

<!-- tutorial-result-start:cell-5 -->
!!! example "运行结果"

    ```text
    Largest fidelity gap between Trotter and oracle: 0.00059
    ```

<!-- tutorial-result-end:cell-5 -->

复苏峰就是初始衰减后保真度的局部极大值。请注意，每个峰值都落在*相同*的 Néel 分支上；此时双生分支的保真度始终可忽略，所以该状态确实回到了出发点。

```python title="Python 单元 6"
peaks = argrelextrema(fatqat_fidelity, np.greater, order=4)[0]
peaks = [p for p in peaks if TIME_GRID[p] > 0.2 and fatqat_fidelity[p] > 0.2]

print("\nRevivals of the |Z2> quench (fatqat Trotter):")
print(f"{'time (us)':>10} {'F(Z2)':>8} {'F(alt)':>8} {'S(t)':>8}")
for p in peaks[:4]:
    print(
        f"{TIME_GRID[p]:>10.2f} {fatqat_fidelity[p]:>8.3f} "
        f"{fatqat_alt[p]:>8.3f} {fatqat_entropy[p]:>8.3f}"
    )

first_peak_time = TIME_GRID[peaks[0]]
first_peak_fidelity = fatqat_fidelity[peaks[0]]
first_peak_entropy = fatqat_entropy[peaks[0]]
print(f"Entropy maximum: {fatqat_entropy.max():.3f}")
```

<!-- tutorial-result-start:cell-6 -->
!!! example "运行结果"

    ```text

    Revivals of the |Z2> quench (fatqat Trotter):
     time (us)    F(Z2)   F(alt)     S(t)
          1.50    0.768    0.000    0.278
          2.95    0.618    0.000    0.496
          4.45    0.482    0.007    0.753
          5.90    0.359    0.005    0.918
    Entropy maximum: 0.938
    ```

<!-- tutorial-result-end:cell-6 -->

## 7. 解读复苏与纠缠熵增长

左图是最关键的结果：在 $t\approx 1.5, 3.0, 4.5$ `us` 附近出现三次复苏，强度一次比一次稍弱，这是 PXP 疤痕的指纹。中图说明这些动力学为何不属于热化：半链熵虽然增长，却会在每次复苏时*下降*，就像该状态短暂地重新想起了如何成为直积态。右图在实空间中讲述同一个故事：Néel 条纹在淬火过程中融化，又在每次保真度达到峰值时部分重组。

```python title="Python 单元 7"
figure, (fid_axis, ent_axis, occ_axis) = plt.subplots(1, 3, figsize=(16, 4))

fid_axis.plot(
    TIME_GRID,
    fatqat_fidelity,
    label="fatqat Trotter",
    linewidth=2,
)
fid_axis.plot(TIME_GRID, oracle_fidelity, "k--", label="exact PXP oracle")
fid_axis.scatter(
    TIME_GRID[peaks],
    fatqat_fidelity[peaks],
    color="C0",
    marker="o",
    zorder=3,
    label="revival peaks",
)
fid_axis.set(
    xlabel="Time (us)",
    ylabel=r"Fidelity $|\langle Z_2|\psi(t)\rangle|^2$",
    ylim=(0, 1.05),
    title="Z2 revival",
)
fid_axis.legend(fontsize="small")

ent_axis.plot(TIME_GRID, fatqat_entropy, label="fatqat Trotter", linewidth=2)
ent_axis.plot(TIME_GRID, oracle_entropy, "k--", label="exact PXP oracle")
for p in peaks[:3]:
    ent_axis.axvline(TIME_GRID[p], color="0.8", linestyle=":", linewidth=1)
ent_axis.set(
    xlabel="Time (us)",
    ylabel=r"Half-chain entropy $S(t)$",
    title="Entanglement growth",
)
ent_axis.legend(fontsize="small")

image = occ_axis.imshow(
    fatqat_occupations.T,
    aspect="auto",
    origin="lower",
    extent=(TIME_GRID[0], TIME_GRID[-1], 0, NUM_SITES),
    cmap="viridis",
)
occ_axis.set(
    xlabel="Time (us)",
    ylabel="Site",
    title=r"Rydberg population $\langle n_i(t)\rangle$",
)
figure.colorbar(image, ax=occ_axis, fraction=0.046, pad=0.04)
figure.tight_layout()
plt.show()
```

<!-- tutorial-result-start:cell-7 -->
!!! example "运行结果"

    ![PXP 复苏保真度、纠缠熵与位点占据](../assets/generated/tutorials/pxp-z2-revival-01.png)

<!-- tutorial-result-end:cell-7 -->
