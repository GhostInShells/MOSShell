"""
MOSS 系统 Signal 地图 — 所有 SignalMeta 的策展入口.

本模块不定义实现, 只做重导出, 让 developer / ghost / channel 从单一入口
即可了解系统内全部信号类型. 新的 SignalMeta 在此注册后, 通过
architecture.py 加入认知地图.

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

from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
)

# -- CellEvent: cell 生命周期变更 → background hint -- #


class CellEventSignalMeta(SignalMeta):
    """Cell 生命周期事件的信号类型.

    由 CellEventNucleus 桥接 mesh.on_event 产生, priority=BACKGROUND —
    不会抢占 attention, 只作为 background hint 进 mindflow buffer.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'cell_event'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.BACKGROUND


__all__ = [
    'InputSignalMeta',
    'NotifySignalMeta',
    'InterruptSignalMeta',
    'CommandSignalMeta',
    'SilentSignalMeta',
    'AudioSignal',
    'CellEventSignalMeta',
]
