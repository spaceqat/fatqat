
# Qiskit


<a id="fatqat.qiskit"></a>

使用 [`FatqatBackend`](#fatqat.qiskit.FatqatBackend) 在 FATQAT 上运行受支持的 Qiskit 电路，或使用 [`circuit_to_program`](#fatqat.qiskit.circuit_to_program) 直接转换电路。Qiskit 需另行安装；不需要 Qiskit Aer。

在 Qiskit 后端工作流程中使用 FATQAT：

```python
from qiskit import QuantumCircuit, generate_preset_pass_manager
from fatqat.qiskit import FatqatBackend

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

backend = FatqatBackend()
pass_manager = generate_preset_pass_manager(backend=backend)
isa_circuit = pass_manager.run(circuit)
job = backend.run(isa_circuit, shots=100, seed_simulator=7)
counts = job.result().get_counts()
```

## 电路转换


[`circuit_to_program`](#fatqat.qiskit.circuit_to_program) 接受已绑定且仅包含 [`build_simulator_target`](#fatqat.qiskit.build_simulator_target) 中指令的电路。其他电路应先转译到该 target。具名寄存器和电路元数据会保留。全局相位存入 `program.metadata`，但不会应用到模拟状态。

测量和重置会被转换，屏障会被丢弃。不受支持的指令、动态控制流、未绑定的指令参数和未绑定的全局相位会引发 [`QiskitConversionError`](#fatqat.qiskit.QiskitConversionError)。

## 后端执行


[`FatqatBackend`](#fatqat.qiskit.FatqatBackend) 默认为 `method="statevector"` 和 `runtime="numpy"`。`method` 接受 [`Simulator`][fatqat.simulator.Simulator] 支持的名称和别名；`runtime` 不区分大小写地接受 `"numpy"` 或 `"numba"`。不受支持的值不会使构造失败，而是让 `run()` 产生错误作业。该后端接受 FATQAT [`NoiseModel`][fatqat.NoiseModel]，而不是 Qiskit Aer 噪声模型。

`run()` 接受一个电路或非空可迭代对象，并支持三个选项：

- `shots`：正 `int`，默认 `1024`；
- `memory`：`bool`，默认 `False`；
- `seed_simulator`：非负 `int` 或 `None`，默认 `None`。

传给 `run()` 的选项会覆盖 `backend.options`。无效选项、空电路可迭代对象或含有非电路值的可迭代对象会在创建作业前引发 [`QiskitBackendError`](#fatqat.qiskit.QiskitBackendError)。如果电路转换或执行之后失败，`run()` 会返回 `ERROR` 作业，[`FatqatJob.result`](#fatqat.qiskit.FatqatJob.result) 会引发 Qiskit 的 `QiskitError`。

结果包含 Qiskit 格式的计数，也适用于多个经典寄存器。`memory=True` 返回与这些计数一致的条目，但不保证 shot 顺序。没有经典比特的电路不含计数或内存数据。状态向量及相关模拟器产物不会包含在 Qiskit 结果中。

## 作业与 provider 辅助类


执行是同步的，因此返回时 [`FatqatJob`](#fatqat.qiskit.FatqatJob) 已处于 `DONE` 或 `ERROR` 状态。[`FatqatProvider`](#fatqat.qiskit.FatqatProvider) 为期望 provider 风格接口的代码创建已配置的 [`FatqatBackend`](#fatqat.qiskit.FatqatBackend) 实例。

## 参考


### 函数 `circuit_to_program(circuit)` { #fatqat.qiskit.circuit_to_program }

将已绑定的静态 Qiskit `QuantumCircuit` 转换为 [`Program`][fatqat.Program]。无法表示的电路会引发 [`QiskitConversionError`](#fatqat.qiskit.QiskitConversionError)；非电路输入会引发 [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError)。

### 类 `FatqatBackend(*, method="statevector", runtime="numpy", noise_model=None, provider=None, name="fatqat_simulator")` { #fatqat.qiskit.FatqatBackend }

由门级 FATQAT 模拟器支持的同步 Qiskit `BackendV2`。

#### 方法 `run(run_input, **run_options)` { #fatqat.qiskit.FatqatBackend.run }

执行一个电路或非空可迭代对象，并返回已完成的 [`FatqatJob`](#fatqat.qiskit.FatqatJob)。

#### 属性 `target` { #fatqat.qiskit.FatqatBackend.target }

用于转译的受支持指令基。

#### 属性 `max_circuits` { #fatqat.qiskit.FatqatBackend.max_circuits }

始终为 `None`；不限制批量大小。

#### 属性 `coupling_map` { #fatqat.qiskit.FatqatBackend.coupling_map }

始终为 `None`；电路不受耦合映射限制。


### 类 `FatqatJob` { #fatqat.qiskit.FatqatJob }

由 [`FatqatBackend.run`](#fatqat.qiskit.FatqatBackend.run) 返回的已完成 Qiskit `JobV1`。

#### 方法 `status()` { #fatqat.qiskit.FatqatJob.status }

返回 Qiskit 的 `DONE` 或 `ERROR` 状态。

#### 方法 `result(timeout=None)` { #fatqat.qiskit.FatqatJob.result }

返回 Qiskit `Result`，或引发 `QiskitError`。接受但忽略 `timeout`。

#### 方法 `submit()` { #fatqat.qiskit.FatqatJob.submit }

由于作业已运行，立即返回。

#### 方法 `cancel()` { #fatqat.qiskit.FatqatJob.cancel }

返回 `False`。

#### 方法 `backend()` { #fatqat.qiskit.FatqatJob.backend }

返回创建此作业的后端。


### 类 `FatqatProvider(**default_backend_kwargs)` { #fatqat.qiskit.FatqatProvider }

创建已配置 [`FatqatBackend`](#fatqat.qiskit.FatqatBackend) 实例的 provider 风格辅助类。

#### 方法 `backends(name=None, *, filters=None, **kwargs)` { #fatqat.qiskit.FatqatProvider.backends }

当 `name` 匹配时，在单元素列表中返回一个新后端；否则返回空列表。`filters` 不产生作用；`kwargs` 中的其他构造选项会覆盖 provider 默认值。

#### 方法 `get_backend(name=None, **kwargs)` { #fatqat.qiskit.FatqatProvider.get_backend }

返回一个新的匹配后端，或引发 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)。


### 函数 `build_simulator_target()` { #fatqat.qiskit.build_simulator_target }

为受支持门基返回一个新的无边界 Qiskit target。

### 异常 `QiskitConversionError` { #fatqat.qiskit.QiskitConversionError }

电路无法转换为 FATQAT 程序。该异常属于 [`fatqat.errors.FatqatError`][fatqat.errors.FatqatError]。

### 异常 `QiskitBackendError` { #fatqat.qiskit.QiskitBackendError }

运行请求在执行前被拒绝。该异常属于 [`fatqat.errors.FatqatError`][fatqat.errors.FatqatError]。
