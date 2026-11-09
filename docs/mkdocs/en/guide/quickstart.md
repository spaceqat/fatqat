# 构建、绘制并运行一个 Program

大约十分钟，你就能完成一个 Bell 对的制备、绘制电路、运行 1,000 次采样，并理解为什么结果只有 `00` 和 `11`。整个计算过程都由同一个 [`Program`][fatqat.Program] 描述。

## 安装 FatQat

FatQat 需要 Python 3.12 或更高版本。如果还没有安装 Python，可以从
[Python 官网](https://www.python.org/downloads/)下载安装。

打开终端：Windows 可以使用 PowerShell；macOS 和 Linux 使用“终端”。下面的
命令在终端中输入，每条输入完整后按回车。

先创建一个学习文件夹：

```sh
mkdir fatqat-learning
cd fatqat-learning
```

创建独立的 Python 环境：

```powershell
# Windows PowerShell
python -m venv .venv
```

```bash
# macOS or Linux
python3 -m venv .venv
```

然后激活它。之后的 `python` 和 `pip` 命令就会使用这个环境：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

安装 FatQat：

```sh
python -m pip install fatqat
```

重新打开终端后，先回到学习文件夹，再运行上面的激活命令；不需要重新安装。

### 从源码安装开发版

如果希望使用最新源码，请克隆仓库：

```bash
git clone https://github.com/spaceqat/fatqat.git
cd fatqat
```

本地项目推荐使用 [uv](https://docs.astral.sh/uv/)，因为它将所选 Python 次版本记录在 `.python-version` 中。这样解释器选择明确，也更容易复现环境：

```bash
uv python pin 3.12
uv venv
uv pip install .
```

将 `.python-version` 保留在项目中，使后续 `uv` 命令选择相同的 Python 次版本。如果更喜欢标准库工具，也可以创建并激活虚拟环境：

```bash
python -m venv .venv

# macOS or Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

然后使用 `pip` 安装 FatQat：

```bash
python -m pip install --upgrade pip
python -m pip install .
```

## 保存并运行 Python 文件

用文本编辑器新建 `first_program.py`，保存在 `fatqat-learning` 中。
把下面的完整代码复制进去；代码在文件中保存，不要当作终端命令输入。
文件名不要变成 `first_program.py.txt`，也不要命名为 `fatqat.py`。

这个小程序把一个量子比特从 `0` 翻转到 `1`，测量后打印结果：

```python
import fatqat as fq
import fatqat.operations as ops

example_program = fq.Program(1, 1)
example_program.add(ops.X, 0)
example_program.measure_all()
example_result = (
    fq.simulator.Simulator(runtime="numpy")
    .run(example_program, shots=100)
    .result()
)
print(example_result.get_counts())
```

保存后，在已激活环境的学习文件夹中执行：

```sh
python first_program.py
```

看到 `{'1': 100}` 就说明程序运行成功。之后修改代码、保存，再运行同一条命令即可。

### 在 Windows 的 VS Code 中运行

如果不想每次输入运行命令，也可以使用 VS Code 的运行按钮：

1. 安装 [VS Code](https://code.visualstudio.com/)，并安装 Microsoft 发布的 Python 扩展。
2. 选择“文件 → 打开文件夹”，打开 `fatqat-learning`，再打开 `first_program.py`。
3. 按 `Ctrl+Shift+P`，选择 **Python: Select Interpreter**，选择该文件夹里的 `.venv`。
   如果没有找到，可以输入解释器路径 `.venv\Scripts\python.exe`。
4. 保存文件，点击编辑区右上角的三角形按钮，选择 **Run Python File in Terminal**。

结果会显示在下方终端。以后修改代码、保存，再点击运行即可。

### 遇到问题时

- **找不到程序文件：** 确认终端在保存 `first_program.py` 的学习文件夹中。
- **提示 `No module named fatqat`：** 确认终端已激活 `.venv`；在 VS Code 中检查选中的解释器。
- **Ubuntu 提示缺少 `venv`：** Ubuntu 24.04 自带的 Python 3.12 可以先执行
  `sudo apt install python3.12-venv`，再创建环境。其他版本的软件包名称可能不同。
- **PowerShell 不允许运行激活脚本：** 可以不激活，改用
  `.\.venv\Scripts\python.exe -m pip install fatqat` 安装，
  再用 `.\.venv\Scripts\python.exe first_program.py` 运行。

如果仍然无法运行，请保留第一条具体报错，并提供操作系统、Python 版本和执行的命令，
到项目的 [Issues](https://github.com/spaceqat/fatqat/issues) 中提问。

下面继续用贝尔态介绍 Program、电路绘图和模拟结果。带 `>>>` 的示例表示 Python
交互式输入；复制到 `.py` 文件时，需要去掉开头的 `>>>` 和续行的 `...`，
也不要复制示例显示的输出。

## 编写一个贝尔态 Program

`Program` 记录量子和经典资源，以及作用于它们的指令。下面的程序将两个量子比特制备为 Bell 态，并测量二者：

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> program = fq.Program(2, 2)
>>> program.add(ops.H, 0)
>>> program.add(ops.CX, (0, 1))
>>> program.measure_all()
```

`Program(2, 2)` 创建两个量子比特和两个经典比特。`H` 使量子比特 0 进入叠加态，`CX` 以量子比特 0 为控制位、量子比特 1 为目标位。最后一条指令把两个量子测量结果写入经典比特。

Program 描述*要做什么*。它尚未选择模拟器，也未执行任何演化。FatQat 还支持扩展：用户可以定义操作，并提供执行这些操作的后端实现。`Put`、`Pair` 和 `Unpair` 等内置操作展示了 `AtomArraySimulator` 如何表达超出标准量子门范围的工作流程。

## 检查 Program

Program 自带电路绘图功能，底层使用 QuTiP-QIP。FatQat 会将程序指令转换为绘图工具可识别的形式。默认返回 Matplotlib 图对象，可直接显示或保存：

```pycon
>>> figure = program.draw()
>>> len(figure.axes)
1
```

![包含 H、受控 X 和两次测量的贝尔态 Program](../assets/generated/guide/quickstart-circuit.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure_all()

    figure, axis = plt.subplots()
    program.draw(ax=axis)
    ```

若需要适合终端的版本，`program.draw("text")` 会返回字符串而不是直接打印。绘图只是所记录指令结构的一种视图，并不会运行 Program。

## 运行

通用模拟器将 Program 作为量子电路演化。`run()` 执行 Program 并返回一个已完成的 [`Job`][fatqat.Job]。调用 `job.result()` 返回包含所请求输出的 [`Result`][fatqat.Result]，或抛出记录的执行错误：

```pycon
>>> backend = fq.simulator.Simulator()
>>> job = backend.run(
...     program,
...     shots=1000,
...     simulation_config={"seed": 7},
... )
>>> result = job.result()
>>> counts = result.get_counts()
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
1000
```

每次采样最终得到 `00` 或 `11`：两个测量位一致，因为量子比特处于纠缠态。测量对态进行采样，所以计数接近但不完全相等。固定种子使这次运行可以复现。`Result.draw()` 为计数提供内置图表，无需自定义 Matplotlib 样式。

![Bell 态测量计数，00 和 11 的柱高近似相等](../assets/generated/guide/quickstart-counts.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure_all()

    result = (
        fq.simulator.Simulator(runtime="numpy")
        .run(program, shots=1000, simulation_config={"seed": 7})
        .result()
    )
    counts = result.get_counts()
    assert sorted(counts) == ["00", "11"]
    assert sum(counts.values()) == 1000

    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    result.draw(
        ax=axis,
        title="Bell-state outcomes",
    )
    ```

## 理解基索引

公共子系统 0 是最高有效位的张量因子。双量子比特基 `|00>`、`|01>`、`|10>`、`|11>` 对应展平索引 0 至 3。因此，`X(q0)|00>` 的非零振幅位于索引 2，`X(q1)|00>` 则位于索引 1。计数字符串采用相同的公共槽位从左到右顺序：将这些态测量到 `(c0, c1)`，分别得到 `"10"` 和 `"01"`。

保留这个 `Program`：后续章节会通过更换后端，探索电路行为、硬件约束与物理动力学。

## 继续

- [用 Program 编写量子计算](program.md)，使用显式 Register、测量、条件、参数和 qudit。
- [选择需要建模的物理细节](execution-models.md)，比较 FatQat 的各条执行路径。
- [在 OpenQASM 和 Qiskit 之间双向转换 Program](interoperability.md)，连接现有的电路工作流。
