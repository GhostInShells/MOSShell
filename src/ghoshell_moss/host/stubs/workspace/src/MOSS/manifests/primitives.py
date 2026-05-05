from ghoshell_moss.core.blueprint.channel_builder import new_command
from ghoshell_moss.core.ctml.shell.primitives import (
    interrupt_command as _interrupt,
    sleep,
    noop,
    observe,
)

# 默认只提供四个原语.
sleep_primitive = new_command(sleep)
noop_primitive = new_command(noop)
observe_primitive = new_command(observe)
interrupt_primitive = _interrupt
