"""
MOSS 系统 Signal 地图 — 所有 SignalMeta 的策展入口.

本模块不定义实现, 只做重导出, 让 developer / ghost / channel 从单一入口
即可了解系统内全部信号类型. 新的 SignalMeta 在此注册后, 通过
architecture.py 加入认知地图.

**纪律**: SignalMeta 的实现随对应 Nucleus 同居 (`core/mindflow/xxx_nucleus.py`).
本文件只 import + __all__. 不要在此就地 class body — 否则等于把两个抽象
撕开放, 违反同伴原则.

目录:
  InputSignalMeta   — 用户输入 (优先级 NOTICE, default mode)
  NotifySignalMeta  — 不丢消息 (优先级 NOTICE, notify mode)
  InterruptSignalMeta — 急停中断 (优先级 FATAL, interrupt mode)
  CommandSignalMeta — 命令执行 (优先级 NOTICE, command_only mode)
  SilentSignalMeta  — 静默聚合 (优先级 NOTICE, silent mode)
  AudioSignal       — 音频感知 (优先级 NOTICE)
  CellEventSignalMeta — Cell 生命周期事件 (优先级 BACKGROUND)
"""
from ghoshell_moss.core.blueprint.mindflow import (
    InputSignalMeta,
)
from ghoshell_moss.core.mindflow import (
    NotifySignalMeta,
    InterruptSignalMeta,
    CommandSignalMeta,
    SilentSignalMeta,
)
from ghoshell_moss.core.mindflow.audio_signal import AudioSignal
from ghoshell_moss.core.mindflow.cell_event_nucleus import (
    CellEventSignalMeta,
    CellTransition,
)

__all__ = [
    'InputSignalMeta',
    'NotifySignalMeta',
    'InterruptSignalMeta',
    'CommandSignalMeta',
    'SilentSignalMeta',
    'AudioSignal',
    'CellEventSignalMeta',
    'CellTransition',
]
