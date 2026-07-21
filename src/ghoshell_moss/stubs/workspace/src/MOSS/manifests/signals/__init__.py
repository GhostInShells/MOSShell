# Signal manifest — signal protocol declarations.
#
# Re-exports the system signal map (ghoshell_moss.signals).  Matrix scans via
# issubclass(obj, SignalMeta), converts each to SignalSchema via to_signal_schema().
#
# Mode extends by: from MOSS.manifests.signals import *
#
# --
# Signal 清单 — 信号协议声明。
# 从系统信号地图 (ghoshell_moss.signals) 重导出，Matrix 扫描自动发现。

from ghoshell_moss.signals import *
