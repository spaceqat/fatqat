# FatQat

FatQat 让你只需将量子计算编写一次，保存为一个 `Program`，随后便可按问题所需的
层次研究它：逻辑线路行为、硬件配置约束，或随时间变化的物理动力学。

下面是完整的线路级工作流：

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure((0, 1), (0, 1))

result = fq.simulator.Simulator().run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()

print(result.get_counts())
```

这段程序制备一个贝尔态。采样结果会随随机种子变化，但只可能出现 `"00"` 和
`"11"`。同一个 `Program` 抽象还可以承载参数、量子多能级系统、经典条件和
直接物理控制。

<div class="grid cards" markdown>

-   :material-play-circle-outline: **[运行你的第一个 Program](guide/quickstart.md)**

    构建并绘制贝尔线路，运行它，再将计数结果绘制成图。

-   :material-transit-connection-variant: **[了解 Program](guide/program.md)**

    在不改变编程模型的前提下，添加寄存器、测量、条件、参数、量子多能级系统及
    混合局域维数。

-   :material-source-branch: **[选择建模层次](guide/execution-models.md)**

    使用同一个 Program，对比通用模拟、硬件配置模拟与哈密顿量级仿真。

-   :material-sine-wave: **[研究硬件行为](guide/hardware-profile-simulation.md)**

    处理原生门、布局、连通性、脉冲控制、泄漏和物理模型。

-   :material-flask-outline: **[学习完整教程](tutorials/index.md)**

    继续学习完整的算法与物理案例，并下载其源代码。

-   :material-format-list-bulleted: **[查阅 API](api/index.md)**

    查找准确的签名、支持的操作、形状、单位和校验规则。

</div>
