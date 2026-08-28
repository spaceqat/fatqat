<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# 自定义噪声实现


本页面向后端作者。大多数应用只需要 FATQAT 的内置噪声类型。使用此扩展 API 定义新的 [`Channel`][fatqat.noise.Channel]，或改变某个后端应用现有类型的方式。模拟器和脉冲仿真器使用不同的映射：模拟器规则返回 Kraus 算符，脉冲仿真器规则返回 Lindblad 算符。

## 选择映射类别


**实现类别**

| 映射 | 规则输入 | 规则输出 |
| --- | --- | --- |
| [`ChannelImplementationMap`][fatqat.noise.ChannelImplementationMap] | `(channel, *, targets)`，其中 `targets` 是程序 [`RegisterRef`][fatqat.RegisterRef] 对象的有序元组 | 作用于合并目标空间的非空 Kraus 矩阵元组 |
| [`LindbladImplementationMap`][fatqat.noise.LindbladImplementationMap] | 针对一个局部物理子系统的 `(channel, *, physical_dimension)` | 非空局部 Lindblad 算符矩阵元组 |

矩阵规则返回一次通道应用的 Kraus 算符。Lindblad 规则返回局部 Lindblad 算符；它们不接收持续时间，因为已用区间由仿真器决定。FATQAT 从不把一个映射类别作为另一个的后备，也从不由概率推导速率。

## 定义自定义噪声类型


自定义噪声类型继承 [`Channel`][fatqat.noise.Channel] 并保存其物理参数。将 `Channel.num_subsystems` 设为固定目标数量；留为 `None` 则使用匹配操作的宽度；也可以在实例数据决定宽度时公开属性。将实例添加到 [`NoiseModel`][fatqat.NoiseModel] 后，应把它们视为不可变对象。

两种映射都使用精确具体类型。若 `type(channel) is MyChannel`，则只有通过 `add(MyChannel, rule)` 注册的规则会匹配。注册 `Channel` 或其他基类不会实现其子类。这样可使后端支持保持显式，并防止新子类悄然继承不兼容的数值规则。

## 注册并复用规则


两个公共映射类具有相同的注册操作：

**注册操作**

| 方法 | 契约 |
| --- | --- |
| [`add`][fatqat.noise.ChannelImplementationMap.add] | 为精确的 `channel_type` 存储可调用对象。再次添加同一类型会替换其规则。FATQAT 立即检查类型和可调用性，在后端使用时检查规则签名和输出。 |
| [`get`][fatqat.noise.ChannelImplementationMap.get] | 返回为精确类型注册的可调用对象，若无则返回 `None`。不会回退到基类。 |
| [`supported_channels`][fatqat.noise.ChannelImplementationMap.supported_channels] | 返回已注册类型的不可变 `frozenset` 快照。 |
| [`copy`][fatqat.noise.ChannelImplementationMap.copy] | 返回一个可独立修改注册项的映射。 |

## 模拟器规则


矩阵规则为匹配操作返回 Kraus 算符：

```python
def rule(channel, *, targets):
    return (kraus_0, kraus_1)
```

如果有序目标的维度为 `d_0, d_1, ...`，其合并维度为 `D = d_0 * d_1 * ...`。结果必须非空，每个元素都必须是形状为 `(D, D)` 的 NumPy 数组。FATQAT 只检查这些结构要求，不会验证完全正性、保迹性、保 Hermitian 性或自定义规则的任何参数约定。

若要保留 FATQAT 的内置模拟器规则，请从 [`default_channel_implementation_map`][fatqat.noise.default_channel_implementation_map] 开始。后端限制参阅[模拟器](backend-support.md#noise-simulator-support)。

### 最小示例


下面的自定义量子比特通道和规则添加比特翻转噪声，同时保留所有内置通道实现：

```python
from dataclasses import dataclass

import numpy as np
import fatqat as fq
import fatqat.operations as ops


@dataclass(frozen=True)
class BitFlip(fq.noise.Channel):
    p: float
    num_subsystems = 1

    def __post_init__(self):
        if (
            isinstance(self.p, bool)
            or not isinstance(self.p, (int, float))
            or not 0.0 <= self.p <= 1.0
        ):
            raise ValueError("p must be a real number in [0, 1]")


def bit_flip_rule(channel, *, targets):
    if targets[0].register.dim != 2:
        raise fq.errors.BackendValidationError("BitFlip requires a qubit")
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return np.sqrt(1 - channel.p) * identity, np.sqrt(channel.p) * x


channel_map = fq.noise.default_channel_implementation_map()
channel_map.add(BitFlip, bit_flip_rule)

noise = fq.NoiseModel()
noise.add(BitFlip(p=0.05), operation=ops.X)
backend = fq.simulator.Simulator(
    method="density_matrix",
    noise=noise,
    channel_implementation_map=channel_map,
)
```

## 脉冲仿真器规则


Lindblad 规则返回局部 Lindblad 算符：

```python
def rule(channel, *, physical_dimension):
    return (lindblad_operator,)
```

结果必须非空，每个元素都必须是形状为 `(physical_dimension, physical_dimension)` 的 NumPy 数组。规则既不接收目标标签，也不接收持续时间。通道必须已使用后端声明的时间单位表示速率或时间参数，规则必须定义这些参数的物理解释。

[`default_lindblad_implementation_map`][fatqat.noise.default_lindblad_implementation_map] 返回标准超导量子比特映射。其他仿真器类别会选择不同的默认值：Atom2 还包含局部退极化，Atom3 则从空映射开始。提供任何显式映射都会替换该类别的默认值。构建替换映射前请参阅[脉冲仿真器](backend-support.md#noise-emulator-support)。

## API


### 模拟器类型


::: fatqat.noise.Channel
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.noise.ChannelImplementation
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
        - "^(?:__call__)$"

::: fatqat.noise.ChannelImplementationMap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.noise.default_channel_implementation_map

### 仿真器映射


::: fatqat.noise.LindbladImplementationMap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.noise.default_lindblad_implementation_map
