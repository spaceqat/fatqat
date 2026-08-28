<!-- 英文对应页由 docs/mkdocs/tools/convert_api.py 从 docs/sphinx/api 生成；此简体中文译文在本文件中维护。 -->

# SampledWaveform


[`SampledWaveform`][fatqat.emulator.SampledWaveform] 描述局部时间网格上的信号。时间使用模型的时间单位，而采样单位以及数值能否为复数则由通道决定。

## 插值


内置脉冲仿真器使用非扭结（not-a-knot）样条插值。两个采样点给出线性曲线，三个给出二次曲线，四个或更多采样点给出三次曲线。

`Atom2LevelEmulator` 在采样区间之外使用零值。`TransmonEmulator` 和 `Atom3LevelEmulator` 则保持最近的端点值，因此若控制在网格之外应处于关闭状态，请把首尾采样值设为零。此行为不会改变操作的持续时间。

样条曲线在采样点之间可能超出给定采样值。如果模型设有振幅限制，仿真器会同时检查插值曲线和采样值。

## 参考


::: fatqat.emulator.SampledWaveform
    options:
      members:
        - "duration"
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
