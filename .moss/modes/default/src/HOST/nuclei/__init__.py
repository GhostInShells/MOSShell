# Nucleus manifest — mindflow perception nucleus declarations.
#
# Define NucleusMeta instances here to declare perception pipelines.
# Matrix scans via isinstance(obj, NucleusMeta) and registers each factory
# into the Mindflow runtime.
#
# --
# Nucleus 清单 — 感知核声明。
# 定义 NucleusMeta 实例声明感知管线，Matrix 扫描自动发现。

from ghoshell_moss.core.mindflow import (
    # interrupt signal handler
    InterruptNucleusMeta,
    # notify model — queued into history on attention contention
    NotifyNucleusMeta,
    # execute a command
    CommandNucleusMeta,
    # silent update — no model wakeup, added to history on attention win
    SilentNucleusMeta,
    # default input signal handler
    InputNucleusMeta,
    # cell lifecycle event → background hint
    CellEventNucleusMeta,
)

# default mode 感知反射弧实例 — 6 个覆盖模型对外的完整感知面:
# input (用户消息) / notify (外部通知) / interrupt (急停) /
# command (命令执行) / silent (静默聚合) / cell_event (cell 生命周期 background hint).
# 全部无参构造使用默认 priority / buffer 参数; mode 层如需覆写, 可在
# 自己的 nuclei/__init__.py 里同名重赋值 (ModeManifests.nuclei() 叠加原则).
input_nucleus = InputNucleusMeta()
notify_nucleus = NotifyNucleusMeta()
interrupt_nucleus = InterruptNucleusMeta()
command_nucleus = CommandNucleusMeta()
silent_nucleus = SilentNucleusMeta()
cell_event_nucleus = CellEventNucleusMeta()
