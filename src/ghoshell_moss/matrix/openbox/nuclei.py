# Openbox nucleus manifest — canonical default mindflow perception nuclei.
#
# Shipped baseline: 6 NucleusMeta instances covering the full perception surface.
# Matrix scans via isinstance(obj, NucleusMeta) and registers each factory
# into the Mindflow runtime.
#
# Project extends by:  from ghoshell_moss.matrix.openbox.nuclei import *
#
# --
# Openbox Nucleus 清单 — 开箱默认感知核（canonical 基线）。
# 6 个 NucleusMeta 实例覆盖完整感知面，Matrix 扫描自动发现并注册到 Mindflow。

from ghoshell_moss.core.mindflow import (
    InterruptNucleusMeta,
    NotifyNucleusMeta,
    CommandNucleusMeta,
    SilentNucleusMeta,
    InputNucleusMeta,
    CellEventNucleusMeta,
)

__all__ = [
    'input_nucleus',
    'notify_nucleus',
    'interrupt_nucleus',
    'command_nucleus',
    'silent_nucleus',
    'cell_event_nucleus',
]

# input (用户消息)
input_nucleus = InputNucleusMeta()

# notify (外部通知)
notify_nucleus = NotifyNucleusMeta()

# interrupt (急停)
interrupt_nucleus = InterruptNucleusMeta()

# command (命令执行)
command_nucleus = CommandNucleusMeta()

# silent (静默聚合)
silent_nucleus = SilentNucleusMeta()

# cell_event (cell 生命周期 background hint)
cell_event_nucleus = CellEventNucleusMeta()
