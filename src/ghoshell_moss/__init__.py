from ghoshell_container import (
    Container,
    IoCContainer,
    get_container,
    set_container,
)

from ghoshell_moss.core import *
from ghoshell_moss.message import *

"""
Ghoshell MOSS 库的 facade, 用来存放最常用的类库引用.

考虑只对外暴露最基础的常用函数. 
"""


def new_chan(name: str, description: str = "", blocking: bool = True) -> PyChannel:
    """
    语法糖, 快速定义一个 Channel.
    """
    return PyChannel(name=name, description=description, blocking=blocking)
