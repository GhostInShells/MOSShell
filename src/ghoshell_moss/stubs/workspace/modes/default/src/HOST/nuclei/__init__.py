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
)
