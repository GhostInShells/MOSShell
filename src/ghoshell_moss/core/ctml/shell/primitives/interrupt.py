from ghoshell_moss.core.concepts.command import PyCommand
from ghoshell_moss.core.concepts.channel import ChannelCtx

__all__ = ['interrupt_command', 'interrupt']


async def interrupt():
    """
    stop all ongoing actions immediately
    """
    runtime = ChannelCtx.runtime()
    if not runtime:
        return
    await runtime.clear_sub_channels()


interrupt_command = PyCommand(interrupt, blocking=True, call_soon=True)
