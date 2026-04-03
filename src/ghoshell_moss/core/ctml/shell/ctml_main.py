from typing import Literal

from ghoshell_moss.core.blueprint import PrimeChannel
from ghoshell_moss.core.concepts.command import PyCommand
from ghoshell_moss.core.py_channel import PyChannel
from .primitives import *

__all__ = [
    "CTMLMainChannel", "create_ctml_main_chan",
    "default_primitives", "default_primitive_map", "experimental_primitives",
]


class CTMLMainChannel(PyChannel):
    """
    ctml 的主 channel.
    """

    pass


default_primitives = [
    wait,
    sample,
    observe,
    sleep,
    clear,
    wait_idle,
    noop,
    branch,
    loop,
]

experimental_primitives = ['wait', 'sample', 'observe', 'interrupt', 'wait_idle']

default_primitive_map: dict[str, PyCommand] = {
    func.__name__: PyCommand(func) for func in default_primitives
}
default_primitive_map['interrupt'] = interrupt_command


def create_ctml_main_chan(
        experimental: bool = True,
        *primitives: str | Literal['*'],
) -> PrimeChannel:
    chan = CTMLMainChannel(
        name="__main__",
        description="CTML Main Channel with primitives",
        blocking=True,
    )
    primitives = list(primitives)
    allow_all = len(primitives) == 0 or '*' in primitives
    if allow_all:
        primitives = list(default_primitive_map.keys())

    # 添加默认原语
    for name in primitives:
        if not experimental and name in experimental_primitives:
            # 跳过实验性质的功能.
            continue
        primitive_command = default_primitive_map.get(name)
        if primitive_command is not None:
            chan.build.add_command(primitive_command)
    return chan
