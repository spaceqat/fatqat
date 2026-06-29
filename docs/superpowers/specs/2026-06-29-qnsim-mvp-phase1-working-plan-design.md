# qnsim MVP Phase 1 — 工作组织方案 (Working Plan)

> 本文档不重复母本设计的语义决定，而是把 **MVP Phase 1 怎么落地实现** 固定下来：
> 工程脚手架、两阶段拆分、每个单元的合同边界、测试节奏。
> 语义合同的 source of truth 仍是设计仓库 `quantum_noisy_simulator_design/`，
> 重点参考 `mvp-minimum-workflow.md` 与 `architecture-program/architecture-program-core-api.md`。

- 日期：2026-06-29
- 目标项目：`qnsim`（位于 `通源量智/qnsim`）
- 范围：MVP **Phase 1** —— 跑通最小 statevector workflow，并配齐 test
- 状态：已与用户确认工作方式，待用户 review 本 spec 后转入实现计划

## 0. 目标闭环（验收标准）

Phase 1 完成的标志是下面这段代码可运行并产出正确结果：

```python
import qnsim as qs

program = qs.Program(2, 2)
program.add(qs.ops.H, 0)
program.add(qs.ops.CZ, (0, 1))
program.add_measurement(0, 0)
program.add_measurement(1, 1)

backend = qs.backends.QubitStatevectorBackend()
job = backend.run(program, shots=1000, result_config=qs.ResultConfig(counts=True))
result = job.result()
print(result.get_counts())
```

公开 gate surface 固定为：

- 无参 gate：`qs.ops.X` `qs.ops.Y` `qs.ops.Z` `qs.ops.H` `qs.ops.T` `qs.ops.CX` `qs.ops.CZ`
- 含参 gate：`qs.ops.RX(theta)` `qs.ops.RY(theta)` `qs.ops.RZ(theta)`
- terminal computational-basis measurement

## 1. 已确认的工作方式决定

| 主题 | 决定 |
|---|---|
| 测试节奏 | **TDD，测试先行**。仍按用户提出的两阶段顺序，但每阶段内部 test-first：先写失败测试，再写实现让它通过。 |
| 工具链 | **uv + src layout + pytest**（uv 0.10.8、Python 3.13.11 均已就绪）。`numpy` 为唯一运行依赖。 |
| 一阶段范围 | **含 `condition=` 归一化，不含执行**。前端完整锁死 `AppliedOperation` 合同（含 condition 规范形），feedforward 的执行留给 backend（Phase 1 backend 并不执行它）。 |
| 一阶段非法输入 | 先抛标准 `ValueError` / `TypeError`；`qnsim.errors` 体系随二阶段 backend 一起建立。 |
| `qs.ops` 实现 | 用子模块 `operations.py` 实现，对外以 `qs.ops` 命名空间暴露。 |

## 2. 工程脚手架（动手前一次性建好）

- 用 `uv` 初始化项目，Python 3.13。
- 目录布局：

  ```
  qnsim/
    pyproject.toml
    src/qnsim/
      __init__.py
    tests/
    docs/
  ```

- `pyproject.toml`：
  - 运行依赖：`numpy`
  - dev 依赖：`pytest`
- 可配国内镜像源（uv 的 `[[tool.uv.index]]` 或环境变量），避免装包卡住。
- 统一测试入口：`uv run pytest`。
- src layout 的作用：测试导入的是已安装的 `qnsim` 包，而不是误导入本地目录，避免打包/导入路径问题。

## 3. 第一阶段 —— 前端对象（test-first）

目标：不依赖任何 backend，就能用 TDD 把前端合同锁死。拆成几个独立可测单元，每个都先写失败测试。

### 3.1 单元拆分与合同边界

| 单元 | 文件 | 锁定的合同 |
|---|---|---|
| Registers | `src/qnsim/registers.py` | `Register` / `QuantumRegister` / `ClassicalRegister`（frozen dataclass，size-first 构造，`name` 关键字）、`RegisterRef`；`reg[i]` 返回 `RegisterRef`，越界抛 `IndexError` |
| Operations | `src/qnsim/operations.py` | `Operation` 基类；按 **class** 区分的 gate（`HGate/TGate/XGate/YGate/ZGate/CXGate/CZGate` 暴露为预置实例，`RX/RY/RZ` 暴露为可初始化 class），经 `qs.ops` 命名空间暴露 |
| Program | `src/qnsim/program.py` | `AppliedOperation` / `Measurement` / `Program`；见 §3.2 |
| 包入口 | `src/qnsim/__init__.py` | 暴露 `qs.Program` / `qs.QuantumRegister` / `qs.ClassicalRegister` / `qs.Measurement` / `qs.ops` |

### 3.2 Program 行为合同

- 构造：`Program(qreg: int | list[QuantumRegister], clreg: int | list[ClassicalRegister] = 0, *, metadata=None)`；
  - 传 `int` 时展开成一个默认 `QuantumRegister` / `ClassicalRegister`。
  - 另有 `Program.registers(*, qreg=[...], clreg=[...])` 显式 register 构造入口。
- 内部状态：`qreg: list[QuantumRegister]`、`creg: list[ClassicalRegister]`、`operations: list[AppliedOperation | Measurement]`、`metadata`。
- `add(op, qreg, *, condition=None)`：
  - `qreg` 单 operand 接受 `int | RegisterRef`，多 operand 必须是 flat `tuple`；**不支持** `add(op, 0, 1)` variadic。
  - 裸 `int` = **全局 flat 索引**（按 register 声明顺序拼接解析）。
  - 校验解析出的 `RegisterRef` 指向 `QuantumRegister`。
  - in-place mutation，返回 `None`，向 `operations` 追加 `AppliedOperation`。
- `add_measurement(qreg, clreg, *, basis=None, observable=None, metadata=None)`：
  - in-place，追加 `Measurement`；`clreg` 接受 `int | RegisterRef`，裸 int 按全局 classical flat 解析。
- `condition=` 归一化（核心，本阶段必须实现）：
  - 公开输入两种糖：单 `(slot, lit)` 或 `((slot, lit), ...)`；判别规则 = `c[0]` 是否为 tuple。
  - 存储规范形唯一：`tuple[ConditionTerm, ...] | None`，`ConditionTerm = (RegisterRef, int)`。
  - 裸 int slot 解析成 `ClassicalRegister` 上的 `RegisterRef`；非 classical ref 拒绝。
  - 多条件语义固定为 `AND`。
  - `add(...)` 复用同一个 `_normalize_condition`。
- `copy()`：返回独立可变副本（值对象不可变，列表浅拷贝即可隔离后续 mutation）。

### 3.3 本阶段明确不做（按 MVP 文档排除）

- `Operation.on(...)`（易用性接口，Phase 1 不进路线）
- feedforward 的**执行**（只做前端归一化，不执行）
- `Reset`、mid-circuit measurement、conditional 执行
- 参数系统、density matrix、noise
- 独立的 `Program.validate()`（结构合法性校验统一留给 backend 入口）

### 3.4 第一阶段测试要点（纯前端、零数值依赖）

- register 取 ref / 越界 / size-first 构造
- 裸 int → 全局 flat 索引解析（多 register 拼接）
- gate 按 class 区分；`qs.ops.X` 是实例、`qs.ops.RX(0.2)` 是带参实例
- `add` / `add_measurement` 的 in-place 与 operations 顺序、类型（`AppliedOperation` vs `Measurement`）
- `condition=` 单条件 / 多条件归一化到统一 AND 规范形；非 classical ref 报错
- 多 operand 必须 tuple、variadic 写法被拒
- MVP 文档里的最小样例断言（`len(program.operations)`、`operations[0].operation.name`、`isinstance(operations[3], Measurement)`）

## 4. 第二阶段 —— backend / engine / 结果（test-first）

按 MVP 文档「实现建议顺序」落地。

### 4.1 单元拆分

| 单元 | 文件 | 内容 |
|---|---|---|
| 异常体系 | `src/qnsim/errors.py` | `QnsimError → BackendValidationError → UnsupportedOperationError`、`ResultFieldUnavailableError` |
| Layout | `src/qnsim/layout.py` | `ResourceLayout`（`system_dims` / `qubit_index` / `clbit_index`）+ backend `resolve_layout(program)`，flat 索引唯一来源；同时是资源-fit 检查落点 |
| Impl map | `src/qnsim/implementation.py` | `MatrixImplementation`、`MatrixRule = Callable[[AppliedOperation], np.ndarray]`、`MatrixImplementationMap`，以及各 gate 的 rule（rule 只出局部矩阵，不算索引；不含参 gate 复用模块级常量矩阵） |
| Engine | `src/qnsim/engine.py` | statevector engine：`apply`（局部矩阵作用到 flat target）/ `measure`（computational basis，投影/采样） |
| 结果 | `src/qnsim/result.py` | `Result` / `ResultConfig`、counts 组装、statevector 返回、`NoMeasurementWarning` |
| Backend + Job | `src/qnsim/backends.py`、`src/qnsim/job.py` | `QubitStatevectorBackend.run() → Job`，`job.result() → Result` |

### 4.2 关键执行合同（来自 MVP 文档，落地时遵守）

- **counts**：key 由 classical register 决定；little-endian（clbit 0 在最右）；多 register 按声明顺序展平；同 slot 取最近写入；未写 clbit 补 `0`。
- **statevector**：无测量时默认附带（确定纯态）；有测量时 opt-in 返回投影态，仅 `shots == 1` 支持，`shots > 1` 同时请求报「Phase 1 暂不支持」。
- **shots**：默认 `1024`；仅产出采样类结果时要求 `> 0`；statevector-only 不检查 shots；需采样却 `<= 0` 时在 `run()` 入口抛 `BackendValidationError`。
- **NoMeasurementWarning**：当且仅当 (1) 产出 counts 且 key 含从未被写过的 clbit，**且** (2) 本次未交付任何态表示 时发出。
- **Job**：eager；状态 `DONE` / `ERROR`；validation 错误在 `run()` 直接 raise，execution 错误标 `ERROR` 并由 `result()` 重抛；只公开 `qnsim.Job`。
- **resolve_layout**（Phase 1 默认实现）：量子/经典各按声明顺序拼接，`system_dims = (2,) * n_qubits`；可重写方法，暂不进稳定公开 API。
- **rule 不算索引**：rule 只产局部 matrix，`target_indices` 由 backend 经 layout 组装进 `MatrixImplementation`。

### 4.3 backend 入口 validation（Phase 1 至少检查）

- register / ref 结构合法性、target/measurement 公开输入形状
- `RegisterRef` 指向的 register subtype 匹配
- operation 是否被支持（由 implementation map 决定，否则 `UnsupportedOperationError`）
- result request 是否被支持
- 程序是否落在该 backend 承诺支持范围（如 measurement 在边界内）

### 4.4 第二阶段测试要点（含数值断言）

- **确定性 case**：`H` 后单 shot 投影态、固定矩阵正确性、Bell-on-CZ 末态、little-endian counts key 顺序、未写 clbit 补 0。
- **统计 case**：大 shots 下 counts 分布在容差内（如 H 后约 50/50）。
- **合同/报错 case**：不支持的 operation → `UnsupportedOperationError`；`shots <= 0` 且需采样 → `BackendValidationError`；`shots > 1` + statevector → 不支持报错；`NoMeasurementWarning` 触发条件。
- 确定性 case 与统计 case 分开组织。

## 5. 测试策略小结

- 每单元 test-first：先写表达合同的失败测试，再写最小实现转绿，再重构。
- 一阶段测试纯前端、零数值依赖，可在没有任何 backend 时全部跑通——这正是先做前端的价值。
- 二阶段数值测试用确定性 + 统计两类分开，避免随机性污染合同断言。
- 统一入口 `uv run pytest`。

## 6. 不在 Phase 1 范围（再次明确）

`Reset`、mid-circuit measurement、conditional/feedforward 执行、noise、density matrix、参数系统、`qs.simulate`、`Experiment`、`Operation.on()` 公开化。
