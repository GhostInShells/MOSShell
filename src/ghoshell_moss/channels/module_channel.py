import importlib
from types import ModuleType
from typing import Optional

from ghoshell_moss.core.blueprint.channel_builder import new_channel, MutableChannel

__all__ = ["new_module_channel"]


def _iter_public_callables(module, include, exclude, respect_all):
    if respect_all:
        names = getattr(module, "__all__", None)
    else:
        names = None

    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]

    for name in names:
        if name.startswith("_"):
            continue
        if include and name not in include:
            continue
        if name in exclude:
            continue
        attr = getattr(module, name, None)
        if attr is not None and callable(attr):
            yield name, attr


def new_module_channel(
    module: str | ModuleType,
    *,
    name: Optional[str] = None,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    respect_all: bool = True,
    description: Optional[str] = None,
) -> MutableChannel:
    """反射一个 Python module, 将其公共函数注册为 Channel 的 command.

    默认 ``respect_all=True``: 如果模块定义了 ``__all__`` 则使用, 否则 fallback 到 ``dir()``.
    设为 ``False`` 则强制忽略 ``__all__``, 始终用 ``dir()`` 发现.

    :param module: module object 或 import path 字符串
    :param name: Channel name, 默认取 ``module.__name__``
    :param include: 白名单, 指定时只取这些函数名
    :param exclude: 黑名单, 排除这些函数名
    :param respect_all: 是否尊重模块的 ``__all__``, 默认 True
    :param description: Channel 描述, 默认自动生成
    """
    if isinstance(module, str):
        module = importlib.import_module(module)

    channel_name = name or module.__name__
    channel_desc = description or f"Reflected module channel for {module.__name__}"

    chan = new_channel(name=channel_name, description=channel_desc)

    exclude_set = set(exclude or [])
    for func_name, func in _iter_public_callables(module, include, exclude_set, respect_all):
        chan.build.command(name=func_name)(func)

    return chan
