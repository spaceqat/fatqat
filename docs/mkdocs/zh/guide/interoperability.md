# 在 OpenQASM、Qiskit 与 Program 之间转换

已经有量子线路？FatQat 可以导入 OpenQASM 3、把 Qiskit `QuantumCircuit`
转换为 [`Program`][fatqat.Program]，也可以作为一个后端出现在 Qiskit 中。

## 选择集成路径

- 如果交换产物应为可移植的源文本，请使用 OpenQASM。
- 如果执行与结果解读应由 FatQat 负责，请转换 Qiskit 线路。
- 如果周围的任务与结果工作流仍应由 Qiskit 负责，请使用 `FatqatBackend`。

## 导入、运行并导出 OpenQASM

OpenQASM 转换器内置于 FatQat，不需要 Qiskit。下面的 OpenQASM 3 源代码
描述了与快速上手中相同的、带测量的贝尔线路：

```pycon
>>> import fatqat as fq
>>> from fatqat.qasm import from_qasm, to_qasm
>>> source = """
... OPENQASM 3.0;
... include "stdgates.inc";
... qubit[2] q;
... bit[2] c;
... h q[0];
... cx q[0], q[1];
... c = measure q;
... """
>>> program = from_qasm(source)
>>> isinstance(program, fq.Program)
True
```

导入的值就是普通 Program。选择后端、运行它，再读取标准 FatQat 结果：

```pycon
>>> counts = (
...     fq.simulator.Simulator()
...     .run(program, shots=100, simulation_config={"seed": 7})
...     .result()
...     .get_counts()
... )
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
100
```

使用 `to_qasm` 将受支持的 Program 导出为 OpenQASM 3：

```pycon
>>> exported = to_qasm(program)
>>> exported.splitlines()[0]
'OPENQASM 3.0;'
>>> "cx q[0], q[1];" in exported
True
```

导出结果是规范化表示，并不保证逐字节复现原文本。寄存器名称可能被处理为安全
或唯一的形式，整寄存器操作可能被展开，而且 Program 元数据不属于 OpenQASM。

OpenQASM 交换目前表示已经绑定、局域维数为二的线路 Program。不受支持的语言
结构或 Program 功能会明确失败，而不会用近似形式替代。支持的语句、特定版本
条件、文件输入与转换错误请参阅 [OpenQASM API 参考](../api/interoperability/openqasm.md)。

## 将 Qiskit 线路转换为 Program

Qiskit 集成是可选功能。请在 FatQat 所在环境中安装 Qiskit；无需 Qiskit Aer：

```bash
python -m pip install qiskit
```

如果想进入 FatQat 边界，并在之后使用其后端和结果模型，请调用
[`circuit_to_program`][fatqat.qiskit.circuit_to_program]：

```python
from qiskit import QuantumCircuit

import fatqat as fq
from fatqat.qiskit import circuit_to_program

circuit = QuantumCircuit(2, 2, name="bell")
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

program = circuit_to_program(circuit)
result = (
    fq.simulator.Simulator()
    .run(program, shots=100, simulation_config={"seed": 7})
    .result()
)
counts = result.get_counts()
```

从 `program` 开始，工作流与直接用 Python 编写 Program 完全相同。尤其是，
你可以选择任意 FatQat 执行模型，并使用 FatQat 的 `Result` 访问器。

## 将 FatQat 用作 Qiskit 后端

如果周围的应用应保持为 Qiskit 工作流，请改用
[`FatqatBackend`][fatqat.qiskit.FatqatBackend]。提交线路之前，先将它转译到
后端目标：

```python
from qiskit import QuantumCircuit, generate_preset_pass_manager
from fatqat.qiskit import FatqatBackend

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

backend = FatqatBackend()
pass_manager = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1,
)
compatible_circuit = pass_manager.run(circuit)

job = backend.run(compatible_circuit, shots=100, seed_simulator=7)
qiskit_result = job.result()
counts = qiskit_result.get_counts()
```

这条路径返回 Qiskit 的任务和结果类型，计数也采用 Qiskit 格式。需要 FatQat
状态、映射或可观测量结果时，请直接转换；Qiskit 后端适配器有意以计数为主。

!!! note "说明"

    适配器接受由其目标基所公布的、已经绑定的静态线路。转译会处理其他受支持的
    Qiskit 门形式；不受支持的动态控制流会被拒绝。`FatqatBackend` 接受的是
    FatQat `NoiseModel`，而不是 Qiskit Aer 噪声模型。

[Qiskit API 参考](../api/interoperability/qiskit.md)记录了准确的目标、转换行为、
运行选项、批处理、内存输出和错误类型。

当输入无法用受支持的 Program 子集表示时，转换器会报告错误，而不会猜测其含义。
