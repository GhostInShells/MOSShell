from typing import Literal

from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.blueprint.channel_builder import new_command
from ghoshell_moss.core.concepts.command import PyCommand
from ghoshell_moss.core.py_channel import PyChannel
from .primitives import *

__all__ = [
    "CTMLMainChannel", "create_ctml_main_chan",
    "default_primitives", "default_primitive_map", "experimental_primitives",
    "inject_system_primitives",
]

from .primitives.thinking import thinking_command


class CTMLMainChannel(PyChannel):
    """
    ctml 的主 channel。当前默认实现。
    新代码推荐使用 ``new_shell_main_channel()`` 获得更干净的空 channel。
    """

    pass


# CTML v1.0.0 KD10: 原语精简 default→5+interrupt/thinking.
# default: 协议必需或 scope 无法表达的; experimental: 与新 scope 语法功能重叠或体系不成熟的.
default_primitives = [
    sleep,
    clear,
    observe,
    noop,
    loop,
    # experimental (功能有效但与 scope 语法重叠或 prompt 风格不成熟, 仅在 experimental=True 时注入):
    wait,
    wait_idle,
    sample,
    branch,
]

experimental_primitives = ['wait', 'wait_idle', 'sample', 'branch']

default_primitive_map: dict[str, PyCommand] = {
    func.__name__: PyCommand(func) for func in default_primitives
}
default_primitive_map['interrupt'] = interrupt_command
default_primitive_map['thinking'] = thinking_command


def inject_system_primitives(main: PrimeChannel, *, extended: bool = False) -> None:
    """
    向 main channel 注入系统原语。这是原语列表的唯一权威来源。

    标准原语 (始终注入)
    扩展原语 (extended=True)

    用法::

        main = new_shell_main_channel()
        inject_system_primitives(main)
        inject_system_primitives(main, extended=True)  # 含实验性原语
    """
    # 标准原语
    main.build.add_command(new_command(sleep))
    main.build.add_command(new_command(clear))
    main.build.add_command(new_command(noop))
    main.build.add_command(new_command(loop))
    main.build.add_command(interrupt_command)  # interrupt 已经是 PyCommand

    if extended:
        main.build.add_command(thinking_command)  # 兼容 thinking 原语.
        main.build.add_command(new_command(observe))
        main.build.add_command(new_command(wait))
        main.build.add_command(new_command(wait_idle))
        main.build.add_command(new_command(sample))
        main.build.add_command(new_command(branch))


def create_ctml_main_chan(
        experimental: bool = True,
        *primitives: str | Literal['*'],
        with_default_primitives: bool = True,
        description: str | None = None,
) -> PrimeChannel:
    """
    创建带默认原语的 main channel。当前默认实现。

    新代码推荐使用 ``new_shell_main_channel()`` + ``inject_system_primitives()``，
    获得更显式的控制。
    """
    chan = CTMLMainChannel(
        name="__main__",
        description=description or "CTML Main Channel with primitives",
        blocking=True,
    )
    if not with_default_primitives:
        return chan
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
