"""animation 轨迹编程 — avatars/<name>/animations.py 编译反射成 ChannelModule.

给 avatar 一个"写代码编排动作"的出口: 模型在 avatars/<name>/animations.py 里写 N 个
``async def`` 函数, 每个函数是一条动画轨迹 (纯代码, 用 await 控时). 驱动用 codex 的
Compiler 编译该文件, 注入 ``get_avatar() -> Avatar`` 与 ``asyncio``, 反射其中的协程
函数, 经 ``channel_builder.new_command`` 变成 command, 打包成一个 ChannelModule 挂到
avatar 主 channel (``with_module``). 同名 module 覆盖 = 热更新原语; 由
``reload_animations`` 命令触发.

每个动画命令 blocking=True —— 它的 await 序列就是时间轨迹, 占主轨; 内部调
``avatar.play`` / ``avatar.param`` 时, 驱动的 idle 让位与 finally 复原自然生效.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

from ghoshell_moss.core.blueprint.channel_builder import new_command
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.concepts.command import Command

from .avatar import Avatar

ANIMATIONS_FILE = "animations.py"
MODULE_NAME = "animations"

# 编译上文: asyncio 直接可用, 模型无需 import.
_PROLOGUE = "import asyncio\n"

# 模板: 模型打开 animations.py (file editor) 时读到的"如何定义"关键 prompt.
STUB = '''"""本形象的自定义动画轨迹.

每个 `async def` 函数 = 一条动画, 自动挂到本形象的主 channel 成为一条命令:
- 函数签名即接口: 参数 = 命令参数, docstring = 命令说明
- 已注入 get_avatar() -> Avatar 与 asyncio, 无需 import
- 控时用 await asyncio.sleep; 动作/参数用 avatar.play / avatar.param
- 编辑后调 reload_animations 热更新 (编译失败保留上一版)

Example:
    async def wave():
        avatar = get_avatar()
        await avatar.play("Tap", 0)
        await asyncio.sleep(0.3)
        await avatar.param("ParamArmLA", 0.8, manual=True)
'''


def _compile(source: str, avatar: Avatar, filename: str) -> ModuleType:
    return Compiler(
        source=_PROLOGUE + source,
        filename=filename,
        modulename=f"avatar_animations_{avatar.name}",
        local_injections={"get_avatar": lambda: avatar},
    ).compiled


def _iter_animations(module: ModuleType):
    """反射模块里的协程函数 (公开且 async def)."""
    for name, obj in module.__dict__.items():
        if name.startswith("_"):
            continue
        if inspect.iscoroutinefunction(obj):
            yield name, obj


class AnimationsModule:
    """ChannelModule (结构化实现): 把 animations.py 的协程函数反射成命令集.

    不需要显式继承 ChannelModule Protocol —— 只要提供 name() + own_commands() 即满足.
    """

    def __init__(self, avatar: Avatar, animations_file: Path):
        self._avatar = avatar
        self._animations_file = animations_file
        source = animations_file.read_text(encoding="utf-8")
        compiled = _compile(source, avatar, str(animations_file))
        self._commands: dict[str, Command] = {
            name: new_command(fn, name=name, blocking=True)
            for name, fn in _iter_animations(compiled)
        }

    def name(self) -> str:
        return MODULE_NAME

    def own_commands(self) -> dict[str, Command]:
        return dict(self._commands)

    async def get_instruction(self) -> str:
        return (
            f"这些是 `{self._avatar.name}` 的自定义动画轨迹, 来自 {ANIMATIONS_FILE}, "
            "每条 async def 是一条命令 (体内纯 Python, 用 get_avatar()/asyncio 控时)。"
            "编辑文件后调 reload_animations 热更新。"
        )


def setup_animations(channel: PrimeChannel, avatar: Avatar, animations_file: Path) -> None:
    """在 avatar 主 channel 上挂动画轨迹: 注册 reload_animations, 有文件则初次挂载.

    ``channel`` 是 PrimeChannel —— 带 ``with_module`` (StatefulChannel) 与 ``build``
    (PrimeChannel), 直接调用, 无需 duck-type。
    """

    async def _reload() -> str:
        if not animations_file.is_file():
            return f"未找到 {ANIMATIONS_FILE} (预期路径: {animations_file})"
        new_module = AnimationsModule(avatar, animations_file)  # 编译失败在此抛出, 保留上一版
        channel.with_module(new_module)
        return f"animations reloaded ({len(new_module.own_commands())} 条)"

    channel.build.command(
        name="reload_animations",
        blocking=True,
        doc=(
            f"重新编译并热加载 {ANIMATIONS_FILE} 里的动画轨迹 (同名 module 覆盖旧命令)。"
            "编译失败抛错并保留上一版。"
        ),
    )(_reload)

    if animations_file.is_file():
        channel.with_module(AnimationsModule(avatar, animations_file))
