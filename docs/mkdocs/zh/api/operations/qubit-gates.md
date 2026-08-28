<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 量子比特门


本页中的每个门都为二维目标（量子比特）定义。[`add`][fatqat.Program.add] 记录操作时不会检查目标维度。程序运行时，后端会检查这一要求及所有设备专用限制。

## 固定门


固定门是可直接使用的值，不得把它们当作函数调用。单量子比特矩阵采用 `|0>, |1>` 基顺序。

**固定单量子比特门**

| 值 | 基态作用 |
| --- | --- |
| [`I`][fatqat.operations.I] | 保持 $\|0\rangle$ 和 $\|1\rangle$ 不变。 |
| [`H`][fatqat.operations.H] | 将 $\|0\rangle$ 映射到 $(\|0\rangle+\|1\rangle)/\sqrt{2}$，将 $\|1\rangle$ 映射到 $(\|0\rangle-\|1\rangle)/\sqrt{2}$。 |
| [`X`][fatqat.operations.X] | 交换 $\|0\rangle$ 和 $\|1\rangle$。 |
| [`Y`][fatqat.operations.Y] | 将 $\|0\rangle$ 映射到 $i\|1\rangle$，将 $\|1\rangle$ 映射到 $-i\|0\rangle$。 |
| [`Z`][fatqat.operations.Z] | 将 $\|1\rangle$ 映射到 $-\|1\rangle$。 |
| [`S`][fatqat.operations.S] | 将 $\|1\rangle$ 映射到 $i\|1\rangle$。 |
| [`Sdg`][fatqat.operations.Sdg] | 将 $\|1\rangle$ 映射到 $-i\|1\rangle$。 |
| [`SX`][fatqat.operations.SX] | 连续应用两次与 X 的作用相同。 |
| [`T`][fatqat.operations.T] | 将 $\|1\rangle$ 映射到 $e^{i\pi/4}\|1\rangle$。 |
| [`Tdg`][fatqat.operations.Tdg] | 将 $\|1\rangle$ 映射到 $e^{-i\pi/4}\|1\rangle$。 |

对于下列多量子比特值，目标顺序严格按表中所示排列。

**固定多量子比特门**

| 值 | 目标顺序 | 基态作用 |
| --- | --- | --- |
| [`CX`][fatqat.operations.CX] | `(control, target)` | 控制位为 `\|1>` 时，对目标位应用 X。 |
| [`CY`][fatqat.operations.CY] | `(control, target)` | 控制位为 `\|1>` 时，对目标位应用 Y。 |
| [`CZ`][fatqat.operations.CZ] | `(control, target)` | 将 `\|11>` 取负。 |
| [`CS`][fatqat.operations.CS] | `(control, target)` | 控制位为 `\|1>` 时，对目标位应用 S。 |
| [`Swap`][fatqat.operations.Swap] | `(target0, target1)` | 交换两个目标的状态。 |
| [`iSwap`][fatqat.operations.iSwap] | `(target0, target1)` | 将 `\|01>` 映射到 `i\|10>`，将 `\|10>` 映射到 `i\|01>`。 |
| [`CCX`][fatqat.operations.CCX] | `(control0, control1, target)` | Toffoli：两个控制位均为 `\|1>` 时应用 X。 |
| [`CSwap`][fatqat.operations.CSwap] | `(control, target0, target1)` | Fredkin：控制位为 `\|1>` 时交换两个目标。 |

### 矩阵定义


这些矩阵作用于列状态向量。单量子比特门的行和列采用基顺序
$(|0\rangle,|1\rangle)$：

$$
\begin{aligned}
I &= \begin{pmatrix}1&0\\0&1\end{pmatrix},
& H &= \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\\[0.5em]
X &= \begin{pmatrix}0&1\\1&0\end{pmatrix},
& Y &= \begin{pmatrix}0&-i\\i&0\end{pmatrix},\\[0.5em]
Z &= \begin{pmatrix}1&0\\0&-1\end{pmatrix},
& S &= \begin{pmatrix}1&0\\0&i\end{pmatrix},\\[0.5em]
\mathrm{Sdg} &= \begin{pmatrix}1&0\\0&-i\end{pmatrix},
& \mathrm{SX} &= \frac{1}{2}\begin{pmatrix}1+i&1-i\\1-i&1+i\end{pmatrix},\\[0.5em]
T &= \begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix},
& \mathrm{Tdg} &= \begin{pmatrix}1&0\\0&e^{-i\pi/4}\end{pmatrix}.
\end{aligned}
$$

对于每个双量子比特矩阵，目标为 $(q_0,q_1)$，行和列采用基顺序
$(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$。第一个操作数
$q_0$ 是局部最高有效位；对于 `CX`、`CY`、`CZ` 和 `CS`，它是控制位，$q_1$ 是目标位。

$$
CX = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&1\\
0&0&1&0
\end{pmatrix}
$$

$$
CY = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&-i\\
0&0&i&0
\end{pmatrix}
$$

$$
CZ = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&-1
\end{pmatrix}
$$

$$
CS = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&i
\end{pmatrix}
$$

对于 `Swap` 和 `iSwap`，同样采用上述基顺序，操作数顺序为 `(target0, target1)`。

$$
\mathrm{Swap} = \begin{pmatrix}
1&0&0&0\\
0&0&1&0\\
0&1&0&0\\
0&0&0&1
\end{pmatrix}
$$

$$
i\mathrm{Swap} = \begin{pmatrix}
1&0&0&0\\
0&0&i&0\\
0&i&0&0\\
0&0&0&1
\end{pmatrix}
$$

对于 `CCX`，操作数顺序为 `(control0, control1, target)`。行和列采用基顺序 $(|000\rangle,|001\rangle,|010\rangle, |011\rangle,|100\rangle,|101\rangle,|110\rangle,|111\rangle)$，第一个操作数为最高有效位：

$$
CCX = \begin{pmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0\\
0&0&0&0&1&0&0&0\\
0&0&0&0&0&1&0&0\\
0&0&0&0&0&0&0&1\\
0&0&0&0&0&0&1&0
\end{pmatrix}
$$

对于 `CSwap`，操作数顺序为 `(control, target0, target1)`。行和列采用同样的三比特基顺序，第一个操作数仍为最高有效位：

$$
\mathrm{CSwap} = \begin{pmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0\\
0&0&0&0&1&0&0&0\\
0&0&0&0&0&0&1&0\\
0&0&0&0&0&1&0&0\\
0&0&0&0&0&0&0&1
\end{pmatrix}
$$

## 参数化门


所有角度均以弧度为单位，且不会归一化。每个角度字段都接受 [`Parameter`][fatqat.Parameter]，供之后通过 [`fatqat.Program.assign_parameters`][fatqat.Program.assign_parameters] 绑定。

**参数化量子比特门**

| 构造函数 | 目标 | 定义 |
| --- | --- | --- |
| [`RX`][fatqat.operations.RX] `(theta)` | 一个标量或一个视图 | 绕 X 轴旋转 `theta`。 |
| [`RY`][fatqat.operations.RY] `(theta)` | 一个标量或一个视图 | 绕 Y 轴旋转 `theta`。 |
| [`RZ`][fatqat.operations.RZ] `(theta)` | 一个标量或一个视图 | 绕 Z 轴旋转 `theta`。 |
| [`Phase`][fatqat.operations.Phase] `(theta)` | 一个标量 | 与 RZ 仅相差全局相位。 |
| [`U`][fatqat.operations.U] `(theta, phi, lam)` | 一个标量 | 使用 Qiskit 参数约定的一般单量子比特门。 |
| [`U1`][fatqat.operations.U1] `(lam)` | 一个标量 | 等价于 `Phase(lam)`。 |
| [`U2`][fatqat.operations.U2] `(phi, lam)` | 一个标量 | 等价于 `U(pi/2, phi, lam)`。 |
| [`U3`][fatqat.operations.U3] `(theta, phi, lam)` | 一个标量 | 矩阵与 `U(theta, phi, lam)` 相同；保留用于 Qiskit 兼容。 |
| [`CPhase`][fatqat.operations.CPhase] `(theta)` | 标量 `(control, target)` | 将 $\|11\rangle$ 乘以 $e^{i\theta}$。 |

### 矩阵定义


这些矩阵作用于列状态向量。下列单量子比特矩阵的行和列采用基顺序
$(|0\rangle,|1\rangle)$。令
$c=\cos(\theta/2)$、$s=\sin(\theta/2)$：

$$
RX(\theta) = \begin{pmatrix}
c&-is\\
-is&c
\end{pmatrix},
\qquad
RY(\theta) = \begin{pmatrix}
c&-s\\
s&c
\end{pmatrix}
$$

$$
RZ(\theta) = \begin{pmatrix}
e^{-i\theta/2}&0\\
0&e^{i\theta/2}
\end{pmatrix},
\qquad
\mathrm{Phase}(\theta) = \begin{pmatrix}
1&0\\
0&e^{i\theta}
\end{pmatrix}
$$

对于 `U`、`U1`、`U2` 和 `U3`，操作数仍采用上述单量子比特基，参数顺序与表中所示构造函数顺序一致：

$$
U(\theta,\phi,\lambda) = \begin{pmatrix}
\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
\end{pmatrix}
$$

$$
U1(\lambda) = \begin{pmatrix}
1&0\\
0&e^{i\lambda}
\end{pmatrix},
\qquad
U2(\phi,\lambda) = \frac{1}{\sqrt{2}}\begin{pmatrix}
1&-e^{i\lambda}\\
e^{i\phi}&e^{i(\phi+\lambda)}
\end{pmatrix}
$$

$$
U3(\theta,\phi,\lambda) = \begin{pmatrix}
\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
\end{pmatrix}
$$

对于 `CPhase`，操作数顺序为 `(control, target)`。行和列采用基顺序 $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$，控制位为局部最高有效位：

$$
\mathrm{CPhase}(\theta) = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&e^{i\theta}
\end{pmatrix}
$$

## API 参考


公共操作属性在[操作概览](../operations.md)中说明。

### 固定值


::: fatqat.operations.I
    options:
      show_attribute_values: false

::: fatqat.operations.H
    options:
      show_attribute_values: false

::: fatqat.operations.X
    options:
      show_attribute_values: false

::: fatqat.operations.Y
    options:
      show_attribute_values: false

::: fatqat.operations.Z
    options:
      show_attribute_values: false

::: fatqat.operations.S
    options:
      show_attribute_values: false

::: fatqat.operations.Sdg
    options:
      show_attribute_values: false

::: fatqat.operations.SX
    options:
      show_attribute_values: false

::: fatqat.operations.T
    options:
      show_attribute_values: false

::: fatqat.operations.Tdg
    options:
      show_attribute_values: false

::: fatqat.operations.CX
    options:
      show_attribute_values: false

::: fatqat.operations.CY
    options:
      show_attribute_values: false

::: fatqat.operations.CZ
    options:
      show_attribute_values: false

::: fatqat.operations.CS
    options:
      show_attribute_values: false

::: fatqat.operations.Swap
    options:
      show_attribute_values: false

::: fatqat.operations.iSwap
    options:
      show_attribute_values: false

::: fatqat.operations.CCX
    options:
      show_attribute_values: false

::: fatqat.operations.CSwap
    options:
      show_attribute_values: false

### 参数化类


::: fatqat.operations.RX
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.RY
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.RZ
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.Phase
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U1
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U2
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.U3
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.CPhase
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
