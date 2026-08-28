<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# PulseControl


[`PulseControl`][fatqat.emulator.PulseControl] 将波形分配给一个物理控制通道。通过模型的 `control` 选择器取得通道，并将结果用于 [PulseOperation](pulse-operation.md) 或[门实现](gate-realization.md)定义。请勿直接构造 [`ControlChannel`][fatqat.emulator.ControlChannel]。

`model.control` 上的方法会在创建通道时检查寻址参数。使用该控制时，仿真器会检查模型类别和具名资源是否兼容，以及波形是否满足模型限制。因此，一个通道可以复用于包含同一资源的另一个兼容模型。内置方法及其单位列于[脉冲控制](index.md)；采样网格之外的插值方式参阅 [SampledWaveform](sampled-waveform.md)。

## 参考


::: fatqat.emulator.PulseControl
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.ControlChannel
    options:
      members: false
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
