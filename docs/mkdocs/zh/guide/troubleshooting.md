# 诊断 Program 工作流

大多数首次运行问题发生在四个边界之一：

```text
write the Program
       |
       v
match the target
       |
       v
configure the run
       |
       v
read the Result
```

请从下面与问题相符的症状开始。通常无需检查 FatQat 的内部执行引擎。

## 编写 Program

???+ question "整数目标含义不明确"

    对于只有一个量子寄存器的 `Program(2)`，使用裸整数很方便。当 Program
    包含多个寄存器后，请使用创建过的寄存器引用：

    ```python
    qubit = fq.QuantumRegister(1, name="qubit")
    qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
    program = fq.Program([qubit, qutrit])

    program.add(ops.X, qubit[0])
    program.add(ops.Shift(1), qutrit[0])
    ```

    这样也能明确每个目标的局域维数。请参阅
    [使用 Program 编写量子计算](program.md)。

??? question "测量报告维数不匹配"

    被测量量子资源与经典目标必须具有相同的局域维数。量子三能级系统需要经典
    三值位，而不是普通比特：

    ```python
    qutrit = fq.QuantumRegister(1, dim=3)
    trit = fq.ClassicalRegister(1, dim=3)
    program = fq.Program([qutrit], [trit])
    program.measure(qutrit[0], trit[0])
    ```

??? question "Program 中仍有未绑定参数"

    绑定会返回一个新的 Program，不会原地修改模板。请保留并提交返回值：

    ```python
    bound_program = template.assign_parameters({theta: 0.4})
    backend.run(bound_program)
    ```

    如果要进行批量绑定，请使用[模拟量子程序](simulation.md)中的扫描工作流。

## 匹配执行目标

???+ question "某个操作可在 Simulator 上运行，但在这里不受支持"

    这通常意味着所选硬件配置或仿真器无法按原样实现该指令。通用模拟器接受
    更广的逻辑操作集合；受约束目标则有意不这样做。

    请检查目标支持的操作，并确认真正想研究哪一项细节：

    - 若要研究不受约束的算法行为，请返回通用模拟器；
    - 若重点是硬件约束，请用该配置的原生操作重写 Program；
    - 只有在确实要扩展目标时，才提供实现或校准。

    FatQat 不会悄悄转译或路由 Program。请参阅
    [选择物理建模的细致程度](execution-models.md)。

??? question "双资源门因布局而失败"

    操作本身可能是原生的，但它的两个物理位置并不连通。请检查或选择
    [`ResourceLayout`][fatqat.ResourceLayout]，并区分有序逻辑目标与其设备
    标签。[在硬件配置上测试 Program](hardware-profile-simulation.md)给出了
    先失败再修正的完整示例。

??? question "仿真器拒绝直接控制"

    每个脉冲通道都来自特定物理模型，并标识该模型的一项资源。请使用与仿真器
    相同的模型构建控制。随后检查波形是否使用模型的时间与控制单位、其取值是否
    合法，以及操作持续时间能否覆盖所有控制。
    [让 Program 进入物理动力学](hamiltonian-emulation.md)会逐项构建这条链。

## 配置运行

??? question "计数每次都会变化"

    计数来自采样。如果想得到更稳定的估计，请增加 `shots`；如果需要复现某个
    特定样本，请设置随机种子：

    ```python
    result = backend.run(
        program,
        shots=1000,
        simulation_config={"seed": 7},
    ).result()
    ```

    固定随机种子有助于调试，但不会让小样本变得更准确。

??? question "含噪或动态运行无法提供状态向量"

    在测量、重置或轨迹噪声之后，一个状态向量只代表一条采样历史。如果需要的
    正是这条单一历史，请用 `shots=1` 请求它。若要得到精确混态，请使用密度
    矩阵模拟。

??? question "第一次 Numba 运行异常缓慢"

    Numba 会延迟编译内核。在测量重复运行时间前先预热工作负载；如果可预测的
    单次延迟更重要，请选择 NumPy。基准测试模式见[性能与扩展](performance.md)。

## 读取 Result

???+ question "Result 访问器称其数据不可用"

    运行只包含所选执行方法生成并且你确实请求的数据。选择访问器前，请先检查
    结果：

    ```python
    print(result.available_data)
    if "statevector" in result.available_data:
        state = result.get_statevector()
    ```

    [解读一次运行](interpret-results.md)会把每种输出与它回答的问题对应起来。

??? question "计数字符串似乎反了"

    显示的计数字符串把经典槽位 0 放在右侧。当局域维数不同，或只是希望获得
    无歧义的程序顺序表示时，请改用 `result.get_counts_as_tuples()`。元组位置
    0 对应经典槽位 0。

## 在不丢失线索的前提下缩小问题

如果上述症状都不匹配，请复制失败的工作流并逐步缩减：

1. 只保留一个寄存器和一个操作；
2. 使用通用 `Simulator` 运行它；
3. 加入所请求的结果字段；
4. 切换到预期的硬件配置或仿真器；
5. 每次恢复一个操作。

行为发生变化的步骤会指出应检查哪个边界。请保留这个最小复现示例——它也是
提交错误报告时最有用的起点。
