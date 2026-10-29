# FatQat

[English](README.md) | 简体中文

FatQat 是一个量子计算工具包，围绕统一的编写接口 `Program` 构建。只需编写一次计算过程，即可选择对底层机器进行多精细的建模。

> **开发状态：** FatQat 正在积极开发中，不同版本之间的接口可能发生变化。需要保证可复现性时，请固定确切的版本号。

| 执行层级 | 适用场景 |
| --- | --- |
| 通用模拟 | 研究逻辑态、采样、可观测量、噪声或参数扫描 |
| 硬件配置模拟 | 检查原生操作、布局、连接关系、容量或原子占据情况 |
| 哈密顿量仿真 | 跟踪脉冲、耦合、泄漏、时序及连续时间噪声 |

不同执行目标接受相同的 `Program` 类型，并通过统一的 `Job`/`Result` 工作流返回结果。每个目标仍会验证自身在物理或数学上能够实现的操作。

## [如果 AI 能写代码，我在哪里最能发挥作用？](https://fatqat.readthedocs.io/en/latest/contributing/contributing-in-the-ai-era/)

**AI 让代码更容易编写。你的问题、判断和审查，让这些代码更有价值。**

提出有用的问题、分享遇到的困难、审查设计、改进说明，或贡献代码。
**[阅读贡献指南 →](https://fatqat.readthedocs.io/en/latest/contributing/contributing-in-the-ai-era/)**

## 安装

FatQat 需要 Python 3.12 或更新版本。安装命令：

```sh
python -m pip install fatqat
```

如需使用开发版，请从源码安装：

```sh
git clone https://github.com/spaceqat/fatqat.git
cd fatqat
python -m pip install .
```

## 运行第一个 Program

下面的贝尔态示例展示了完整的量子电路工作流：

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

print(result.get_counts())
```

结果只会出现 `00` 和 `11`：两个量子比特处于纠缠态，因此测得的两个比特相同。[快速入门](https://fatqat.readthedocs.io/en/latest/guide/quickstart/)介绍了如何绘制并运行这个 Program，再将计数结果绘制成图。

## 扩展同一套编写模型

`Program` 记录寄存器和有序指令，支持量子门、测量、重置、经典条件、可复用参数、逻辑 qudit（多能级量子系统）、混合局部维度、电路绘图和直接物理控制。这些功能使用同一套模型，无需分别使用电路语言和脉冲语言。

[Program 指南](https://fatqat.readthedocs.io/en/latest/guide/program/)循序渐进地介绍这些概念。[选择物理建模的精细程度](https://fatqat.readthedocs.io/en/latest/guide/execution-models/)则展示了如何在三个执行层级运行同一个旋转操作，而无需修改它。

接下来可以阅读：

- [模拟量子程序](https://fatqat.readthedocs.io/en/latest/guide/simulation/)：量子态、采样和参数扫描。
- [解读运行结果](https://fatqat.readthedocs.io/en/latest/guide/interpret-results/)：计数、量子态、映射和可观测量。
- [比较理想执行与含噪执行](https://fatqat.readthedocs.io/en/latest/guide/ideal-and-noisy/)。
- [衡量性能与扩展性](https://fatqat.readthedocs.io/en/latest/guide/performance/)。
- [测试硬件配置](https://fatqat.readthedocs.io/en/latest/guide/hardware-profile-simulation/)。
- [从 Program 到物理动力学](https://fatqat.readthedocs.io/en/latest/guide/hamiltonian-emulation/)，然后继续了解 transmon 或中性原子工作流。
- [连接 OpenQASM 和 Qiskit](https://fatqat.readthedocs.io/en/latest/guide/interoperability/)。

[教程集](https://fatqat.readthedocs.io/en/latest/tutorials/)提供了更完整的算法和物理案例。[API 参考](https://fatqat.readthedocs.io/en/latest/api/)给出了确切的函数签名、支持的操作、形状、单位和验证约定。

发布说明见[更新日志](https://github.com/spaceqat/fatqat/blob/main/CHANGELOG.md)。上述文档站链接目前指向英文内容。

## 开发

进行基本的本地开发时，安装源码及核心测试依赖：

```sh
python -m pip install --upgrade pip
python -m pip install --editable . --group dev
python -m pytest
```

准备贡献前，请安装完整的测试和代码检查环境：

```sh
python -m pip install --editable . --group test-full --group lint
```

`test-full` 包含 `dev` 依赖和可选的 Qiskit 集成依赖。`lint` 添加 Black 和 Pylint。

提交修改前，请阅读[贡献指南](CONTRIBUTING.zh-CN.md)，包括 AI 辅助工作的相关政策。项目允许使用 AI 工具，但每位贡献者必须理解自己的工作、对其负责，并亲自主导提交和相关项目交流。

修改文档时，请遵循[固定版本的环境配置与构建流程](https://github.com/spaceqat/fatqat/blob/main/docs/mkdocs/README.md)。

主要目录：

- `src/fatqat/`：软件包源码。
- `tests/`：面向行为的测试套件。
- `docs/mkdocs/`：Material 用户指南、可执行教程和 API 参考。
