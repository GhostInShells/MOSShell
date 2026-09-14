"""avatar_node — Live2D 形象驱动框架.

形象作者 (写 `avatars/<name>/channel.py` 的模型) 只需要这个包的公开面:

    from avatar_node import Avatar

    async def build(avatar: Avatar) -> Channel:
        chan = new_channel(name="hiyori", description="...")

        @chan.build.command()
        async def look(x: float = 0.0, y: float = 0.0) -> None:
            '''让眼睛转向 (x, y), 取值 -1 到 1'''
            avatar.params({"ParamEyeBallX": x, "ParamEyeBallY": y})

        return chan

驱动负责: 发现套件, 解析模型元数据, 同源提供页面与资产, 把事件送到页面.
作者负责: 这个形象对外是什么命令面.
"""

from .avatar import Avatar
from .bridge import AvatarBridge
from .cubism import Group, ModelSpec, Param
from .discovery import (
    AvatarKit,
    AvatarNotFoundError,
    discover,
    load_channel,
    load_spec,
    model_url,
    select,
)
from .lexicon import group_slug, param_ident

__all__ = [
    "Avatar",
    "AvatarBridge",
    "AvatarKit",
    "AvatarNotFoundError",
    "Group",
    "ModelSpec",
    "Param",
    "discover",
    "group_slug",
    "load_channel",
    "load_spec",
    "model_url",
    "param_ident",
    "select",
]
