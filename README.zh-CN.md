<div align="center">

<img src="docs/mkdocs/en/assets/images/space-qat-logo.png" alt="" width="120">

# FatQat

**同一个量子程序，从算法到硬件**

[![Tests](https://github.com/spaceqat/fatqat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/spaceqat/fatqat/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/fatqat)](https://pypi.org/project/fatqat/)
[![Python](https://img.shields.io/pypi/pyversions/fatqat)](https://pypi.org/project/fatqat/)
[![Documentation](https://img.shields.io/readthedocs/fatqat)](https://fatqat.readthedocs.io/zh-cn/latest/)
[![License](https://img.shields.io/pypi/l/fatqat)](https://github.com/spaceqat/fatqat/blob/main/LICENSE)

[快速入门](https://fatqat.readthedocs.io/zh-cn/latest/guide/quickstart/) · [文档](https://fatqat.readthedocs.io/zh-cn/latest/) · [应用案例](https://fatqat.readthedocs.io/zh-cn/latest/tutorials/)

[English](README.md) | 简体中文 | [한국어](README-KR.md)

</div>

FatQat 是一个用于编写、编译和运行量子程序的 Python 工具包。量子程序（`Program`）只需编写一次，
就能通过同一套接口在各类后端上运行——从理想模拟器到设备的物理模型。

- **算法开发者**可以在接近真实设备的条件下验证算法，为在硬件上运行做好准备。
- **硬件工程师**可以模拟设备对控制、耦合和噪声的响应，找出限制性能的因素。
- **学生**可以借助循序渐进的示例和应用案例动手学习量子计算，从第一个量子线路一直深入到真实设备的物理。

现有后端提供三种层次的物理细节：

| 模型 | 回答的问题 |
| --- | --- |
| 通用模拟 | 算法能否按预期工作？噪声会如何改变结果？ |
| 硬件约束模拟 | 程序是否满足设备对原生门、连通性和原子占据的要求？ |
| 哈密顿量仿真 | 脉冲作用期间会发生什么：泄漏、耦合与时序？ |

`Program` 不只是量子门线路，它还可以包含：

- 作用在量子比特、qudit 或光学模式上的量子门、线路中间测量、前馈和参数；
- 设备操作，例如原子装载、原子配对和原子丢失；
- 直接编写的控制脉冲，可与经过校准的量子门混合使用。

FatQat 还能将逻辑线路编译到超导和中性原子平台，并支持与 OpenQASM 和 Qiskit 相互转换线路。

## 安装

FatQat 需要 Python 3.12 或更高版本：

```sh
python -m pip install fatqat
```

## 运行第一个 Program

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(program, shots=1000).result()
print(result.get_counts())  # 只有 '00' 和 '11'
```

## 了解更多

- [快速入门](https://fatqat.readthedocs.io/zh-cn/latest/guide/quickstart/)：
  编写、绘制并运行第一个 Program。
- [用户指南](https://fatqat.readthedocs.io/zh-cn/latest/guide/)：
  按任务介绍模拟、硬件约束和脉冲动力学。
- [应用案例](https://fatqat.readthedocs.io/zh-cn/latest/tutorials/)：
  量子纠错、量子模拟、优化和脉冲控制。
- [API 参考](https://fatqat.readthedocs.io/zh-cn/latest/api/)：
  精确的签名、单位和验证规则。

各版本的变更见[更新日志](https://github.com/spaceqat/fatqat/blob/main/CHANGELOG.md)。

## 参与贡献

AI 让编写代码变得更容易。你的问题、使用场景和评审，能帮助决定哪些代码值得编写。
[了解你可以如何发挥作用](https://fatqat.readthedocs.io/zh-cn/latest/contributing/contributing-in-the-ai-era/)，
然后阅读[为 FatQat 做贡献](CONTRIBUTING.zh-CN.md)，了解开发环境配置和 AI 辅助工作的政策。

FatQat 采用 [Apache License 2.0](https://github.com/spaceqat/fatqat/blob/main/LICENSE) 许可证。
