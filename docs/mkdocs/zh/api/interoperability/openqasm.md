
# OpenQASM


<a id="fatqat.qasm"></a>

使用 [`from_qasm`][fatqat.qasm.from_qasm] 或 [`from_qasm_file`][fatqat.qasm.from_qasm_file] 将 OpenQASM 2 或 3 导入 [`Program`][fatqat.Program]，使用 [`to_qasm`][fatqat.qasm.to_qasm] 导出程序。这些函数不需要 Qiskit。现有代码可以继续使用 `qasm_to_program` 和 `program_to_qasm`。

```python
from fatqat.qasm import from_qasm, to_qasm

program = from_qasm(
    "OPENQASM 3.0; "
    'include "stdgates.inc"; '
    "qubit[2] q; h q[0]; cx q[0], q[1];"
)
source = to_qasm(program)  # OpenQASM 3.0 by default
```

## 导入支持


导入的量子和经典寄存器维度均为 2。FATQAT 支持标量操作数、大小相等的整寄存器操作、测量、重置、局部 `gate` 定义，以及 [`from_qasm`][fatqat.qasm.from_qasm] 列出的内置项。屏障会被忽略。条件可以通过整寄存器相等比较或多个比特比较的逻辑与来控制门或重置。

`include` 语句不会加载文件。通常由 include 提供的门必须已经是内置门，或在局部 `gate` 块中定义。其他控制流、门修饰符、声明和经典条件会引发 [`QASMTranspileError`][fatqat.qasm.QASMTranspileError]。

## 导出支持


[`to_qasm`][fatqat.qasm.to_qasm] 默认生成 OpenQASM 3。程序必须已完全绑定。只有当每个条件都比较单个经典寄存器中的每一位时，才能使用 `version=2`。

导出要求每个寄存器都满足 `dim == 2`，并且操作目标为标量。[`to_qasm`][fatqat.qasm.to_qasm] 列出了受支持门和二维简化。为避免冲突，寄存器名称可能发生变化；程序元数据会被省略。

屏障、直接脉冲控制、[`RegisterView`](../registers.md#fatqat.RegisterView) 目标以及没有 QASM 表示的操作会引发 [`QasmExportError`][fatqat.qasm.QasmExportError]。

## 错误


捕获 [`fatqat.errors.FatqatError`][fatqat.errors.FatqatError] 可以统一处理两类转换错误。若要分别处理转换器拒绝和无法表示程序的 [`QasmExportError`][fatqat.qasm.QasmExportError]，请捕获 [`QASMTranspileError`][fatqat.qasm.QASMTranspileError]。为保持兼容，[`QASMTranspileError`][fatqat.qasm.QASMTranspileError] 同时也是 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)。

[`from_qasm_file`][fatqat.qasm.from_qasm_file] 的文件 I/O 和文本解码故障使用底层 Python 异常。

## 参考


::: fatqat.qasm.from_qasm

::: fatqat.qasm.from_qasm_file

::: fatqat.qasm.to_qasm

::: fatqat.qasm.QASMTranspileError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.qasm.QasmExportError
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
