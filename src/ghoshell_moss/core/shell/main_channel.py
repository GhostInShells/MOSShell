from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.py_channel import PyChannel

__all__ = ["MainChannel"]


class MainChannel(PyChannel):
    pass


async def react(self_instruction: str = "") -> str:
    """
    观察迄今发生的事情, 并触发你下一轮思考.
    :param self_instruction: 可指定下一轮要求自己看到的提示. 通常不用填写.
    """
    if self_instruction:
        return f"{self_instruction}"
    return "do observe and react"


def create_main_channel() -> Channel:
    chan = MainChannel(
        name="",
        description="",
        block=True,
    )

    chan.build.command()(react)

    return chan


# primitive.py 原语定义成command
# wait_done 原语
# shell 调用自己，stop，避免循环
#   shell等待所有的命令执行完，但是避免 wait_done
