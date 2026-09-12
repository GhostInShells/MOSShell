"""Dolores 的 ghost 根 channel — 反身性控制面 | 集成 | alpha

根节点在 Dolores 自身的环境里组装: 闭包持有 ghost 的资源 (共享的 GroundSet, ghost home),
把子 channel 挂成静态子节点. 由 ``Ghost.channel()`` 返回, runtime 以 ``ghost`` 名注册,
CTML 路径即 ``ghost.*``.

Example:
    from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

    chan = build_dolores_channel(groundset=gs, workspace_root=ghost_home)
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.channels.ground_channel import new_ground_channel
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.ground import GroundSet

__all__ = ["build_dolores_channel"]


def build_dolores_channel(
    *,
    groundset: GroundSet,
    workspace_root: str | Path,
    name: str = "ghost",
    description: str | None = None,
) -> MutableChannel:
    """组装 Dolores 的 ghost 根 channel.

    :param groundset: ghost 持有的 GroundSet — 注入给 ground 子 channel 共享, 不新建
        (两个 set 压同一目录 = 两个 snapshot owner).
    :param workspace_root: 子 channel 相对路径解析基点 (ghost home).
    :param name: channel 名 (runtime 以 ``ghost`` 为键注册, 同名保持一致).
    :param description: 覆盖默认描述.
    """
    if description is None:
        description = "Dolores 的反身性控制面 — 子 channel 是它自身的器官."

    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        return (
            "## Ghost (反身性)\n"
            "本 channel 是你自身的控制面: 它的子 channel 是你自己的器官, 不是外部工具.\n"
            "器官的状态与内容随认知场刷新; 命令签名由 interface 反射, 这里不复述."
        )

    chan.import_channels(
        new_ground_channel(
            groundset,
            workspace_root=workspace_root,
            render_root=False,
            name="ground",
            description="认知场 (GROUND.md) — 法链跨 compact 存活, open/close 挂场为子 channel.",
        )
    )
    return chan
