
# 量子多能级门


这些门为局部维度 `d >= 2` 定义；表中给出了额外的维度规则。[`add`][fatqat.Program.add] 记录操作，程序运行时由后端检查支持情况。整数 `power` 会对相应目标维度取模，因此负值和超出范围的值均有效。

**量子多能级门**

| 值或构造函数 | 目标与约束 | 基态作用 |
| --- | --- | --- |
| [`Shift`][fatqat.operations.Shift] `(power)` | 一个标量；任意 `d >= 2` | `\|k> -> \|(k + power) mod d>`。`d=2` 时 `Shift(1)` 即 X。 |
| [`Clock`][fatqat.operations.Clock] `(power)` | 一个标量；任意 `d >= 2` | `\|k> -> omega**(k*power)\|k>`，其中 `omega=exp(2*pi*i/d)`。`d=2` 时 `Clock(1)` 即 Z。 |
| [`Sum`][fatqat.operations.Sum] | 维度相同的 `(control, target)` | `\|i,j> -> \|i,(i+j) mod d>`。`d=2` 时即 CX。 |
| [`SwapLevels`][fatqat.operations.SwapLevels] `(j, k)` | 一个标量；`0 <= j,k < d` 且 `j != k` | 交换 `\|j>` 与 `\|k>`，其他能级保持不变。 |
| [`Fourier`][fatqat.operations.Fourier] | 一个标量；任意 `d >= 2` | `\|j> -> sum(exp(2*pi*i*j*k/d)\|k>) / sqrt(d)`。`d=2` 时即 H。 |
| [`InverseFourier`][fatqat.operations.InverseFourier] | 一个标量；任意 `d >= 2` | `Fourier` 的共轭转置；使用负指数。 |
| [`SubspaceRX`][fatqat.operations.SubspaceRX] `(theta, (j, k))` | 一个标量；两个不同且在范围内的能级 | 令 `c=cos(theta/2)`、`s=sin(theta/2)`：`\|j> -> c\|j>-i*s\|k>`，`\|k> -> -i*s\|j>+c\|k>`。 |
| [`SubspaceRY`][fatqat.operations.SubspaceRY] `(theta, (j, k))` | 一个标量；两个不同且在范围内的能级 | `\|j> -> c\|j>+s\|k>`，`\|k> -> -s\|j>+c\|k>`。反转 `(j, k)` 会反转旋转方向。 |
| [`SubspaceRZ`][fatqat.operations.SubspaceRZ] `(theta, (j, k))` | 一个标量；两个不同且在范围内的能级 | `\|j>` 获得 `exp(-i*theta/2)`，`\|k>` 获得 `exp(i*theta/2)`。反转该对会反转旋转方向。 |
| [`CClock`][fatqat.operations.CClock] `(power)` | `(control, target)`；维度可以不同 | `\|i,j>` 获得 `omega**(i*j*power)`，其中使用目标的 `omega=exp(2*pi*i/d_target)`。两个量子比特且 power 为 1 时即 CZ。 |

能级对必须包含两个不同的非负整数。构造时检查相等和负值，添加操作时检查目标维度。`Sum` 要求控制和目标维度相同；程序运行时，后端会拒绝维度不匹配。

## 矩阵定义


以下矩阵作用于列向量。对于单量子多能级门，行和列采用计算基顺序
$\lvert 0\rangle,\lvert 1\rangle,\ldots,\lvert d-1\rangle$。

### Shift 与 Clock


对于 `Shift(power=p)` 和 `Clock(power=p)`，令
$\omega_d=\exp(2\pi i/d)$。它们的一般算符为

$$
X_d^p
= \sum_{k=0}^{d-1}
  \left\lvert (k+p)\bmod d \right\rangle\!\left\langle k\right\rvert,
\qquad
Z_d^p
= \sum_{k=0}^{d-1}
  \omega_d^{pk}\left\lvert k\right\rangle\!\left\langle k\right\rvert.
$$

### Sum


`Sum` 的操作数为 `(control, target)`。当二者维度同为 `d` 时，控制位是局部最高有效因子：

$$
\operatorname{SUM}_d
= \sum_{i,j=0}^{d-1}
  \left\lvert i,(i+j)\bmod d\right\rangle
  \!\left\langle i,j\right\rvert.
$$

### SwapLevels


对于不同能级 $j$ 和 $k$，一般算符为

$$
S_{j,k}
= I - \lvert j\rangle\!\langle j\rvert
    - \lvert k\rangle\!\langle k\rvert
    + \lvert j\rangle\!\langle k\rvert
    + \lvert k\rangle\!\langle j\rvert.
$$

### Fourier 变换


令 $\omega_d=\exp(2\pi i/d)$，`Fourier` 和 `InverseFourier` 为

$$
F_d
= \frac{1}{\sqrt d}\sum_{j,k=0}^{d-1}
  \omega_d^{jk}\lvert k\rangle\!\langle j\rvert,
\qquad
F_d^{-1}=F_d^\dagger
= \frac{1}{\sqrt d}\sum_{j,k=0}^{d-1}
  \omega_d^{-jk}\lvert k\rangle\!\langle j\rvert.
$$

### 子空间旋转


令 $c=\cos(\theta/2)$、$s=\sin(\theta/2)$。对于有序能级对 `subspace=(j, k)`，一般算符为

$$
\begin{aligned}
R_X^{(j,k)}(\theta)
&= I + (c-1)(\lvert j\rangle\!\langle j\rvert
                +\lvert k\rangle\!\langle k\rvert)
   -is(\lvert j\rangle\!\langle k\rvert
                +\lvert k\rangle\!\langle j\rvert), \\
R_Y^{(j,k)}(\theta)
&= I + (c-1)(\lvert j\rangle\!\langle j\rvert
                +\lvert k\rangle\!\langle k\rvert)
   -s\lvert j\rangle\!\langle k\rvert
   +s\lvert k\rangle\!\langle j\rvert, \\
R_Z^{(j,k)}(\theta)
&= I + (e^{-i\theta/2}-1)\lvert j\rangle\!\langle j\rvert
   +(e^{i\theta/2}-1)\lvert k\rangle\!\langle k\rvert.
\end{aligned}
$$

### CClock


`CClock(power=p)` 的操作数为 `(control, target)`。若二者维度分别为 $d_c$ 和 $d_t$，则控制位是局部最高有效因子，$\omega_t=\exp(2\pi i/d_t)$，且

$$
\operatorname{CClock}_{d_c,d_t}^{(p)}
= \sum_{i=0}^{d_c-1}\sum_{j=0}^{d_t-1}
  \omega_t^{ijp}\lvert i,j\rangle\!\langle i,j\rvert.
$$

## API 参考


公共操作属性在[操作概览](../operations.md)中说明。

::: fatqat.operations.Shift
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.Clock
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.Sum
    options:
      show_attribute_values: false

::: fatqat.operations.SwapLevels
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.Fourier
    options:
      show_attribute_values: false

::: fatqat.operations.InverseFourier
    options:
      show_attribute_values: false

::: fatqat.operations.SubspaceRX
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.SubspaceRY
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.SubspaceRZ
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

::: fatqat.operations.CClock
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
