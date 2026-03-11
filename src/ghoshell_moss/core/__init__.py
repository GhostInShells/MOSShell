from .concepts import *
from .duplex import (
    Connection,
    ConnectionClosedError,
    ConnectionNotAvailable,
    DuplexChannelRuntime,
    DuplexChannelProvider,
    DuplexChannelProxy,
)
from .duplex.protocol import *
from .py_channel import PyChannel, PyChannelRuntime, PyChannelBuilder
from .ctml.shell import CTMLShell, create_ctml_main_chan, new_ctml_shell


def new_channel(
    name: str,
    description: str = "",
    *,
    blocking: bool = True,
) -> MutableChannel:
    """
    创建 MutableChannel.
    """
    return PyChannel(name=name, description=description, blocking=blocking)
