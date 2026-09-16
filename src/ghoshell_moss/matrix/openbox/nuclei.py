# Openbox nucleus manifest — canonical default mindflow perception nuclei.
#
# Shipped baseline: 7 NucleusMeta instances covering the full perception surface.
# Matrix scans via isinstance(obj, NucleusMeta) and registers each factory
# into the Mindflow runtime.
#
# Project extends by:  from ghoshell_moss.matrix.openbox.nuclei import *

from ghoshell_moss.core.mindflow import (
    InterruptNucleusMeta,
    NotifyNucleusMeta,
    CommandNucleusMeta,
    AsideNucleusMeta,
    InputNucleusMeta,
    CellEventNucleusMeta,
    ListenerNucleusMeta,
)

__all__ = [
    'input_nucleus',
    'notify_nucleus',
    'interrupt_nucleus',
    'command_nucleus',
    'aside_nucleus',
    'cell_event_nucleus',
    'listener_nucleus',
]

# input — user message (aggregate buffer, turn toward the user)
input_nucleus = InputNucleusMeta()

# notify — must-not-lose message (buffer on loss)
notify_nucleus = NotifyNucleusMeta()

# interrupt — stop now (take attention then drop)
interrupt_nucleus = InterruptNucleusMeta()

# command — execute logos directly
command_nucleus = CommandNucleusMeta()

# aside — notice without interrupting (buffer without attention)
aside_nucleus = AsideNucleusMeta()

# cell_event — cell lifecycle background hint
cell_event_nucleus = CellEventNucleusMeta()

listener_nucleus = ListenerNucleusMeta()