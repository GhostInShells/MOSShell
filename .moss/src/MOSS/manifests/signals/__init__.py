# Signal manifest — signal protocol declarations.
#
# Define SignalMeta subclasses to declare typed signals.  Matrix scans via
# issubclass(obj, SignalMeta), converts each to SignalSchema via to_signal_schema().
#
# Mode extends by: from MOSS.manifests.signals import *
#
# --
# Signal 清单 — 信号协议声明。
# 用 SignalMeta 子类声明类型化信号，Matrix 扫描自动发现。

from ghoshell_moss.core.mindflow import (
    InputSignalMeta,
    NotifySignalMeta,
    InterruptSignalMeta,
    CommandSignalMeta,
    SilentSignalMeta,
)
