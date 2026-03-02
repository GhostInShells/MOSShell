from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.py_channel import PyChannel
from .primitives import *

__all__ = ["CTMLMainChannel", "create_ctml_main_chan"]


class CTMLMainChannel(PyChannel):
    """
    ctml 的主 channel.
    """
    pass


def create_ctml_main_chan() -> Channel:
    chan = CTMLMainChannel(
        name="__main__",
        description="CTML Main Channel with primitives",
        blocking=True,
    )

    # wait 原语
    chan.build.command()(wait)
    # sleep 原语
    chan.build.command()(sleep)
    # clear 原语
    chan.build.command()(clear)
    # wait idle 原语.
    chan.build.command()(wait_idle)
    chan.build.command()(noop)
    chan.build.command()(observe)
    chan.build.command()(branch)
    chan.build.command()(loop)
    chan.build.add_command(interrupt_command)

    return chan

# primitive.py 原语定义成command
# wait_done 原语
# shell 调用自己，stop，避免循环
#   shell等待所有的命令执行完，但是避免 wait_done
