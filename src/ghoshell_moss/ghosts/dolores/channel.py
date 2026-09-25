"""Dolores 的 ghost 根 channel — 反身性控制面 | 集成 | alpha

根节点在 Dolores 自身的环境里组装: 闭包持有 ghost 的资源 (共享的 GroundSet, memento
manager, ghost home), 把子 channel 挂成静态子节点. 由 ``Ghost.channel()`` 返回, runtime
以 ``ghost`` 名注册, CTML 路径即 ``ghost.*``.

子 channel 都是 ghost 自己的器官 —— ``ghost.ground`` 是它的认知场, ``ghost.memento``
是它的记忆.

Example:
    from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

    chan = build_dolores_channel(groundset=gs, memento_manager=mm, workspace_root=ghost_home)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ghoshell_moss.channels.frame_channel import Frame, new_frame_channel
from ghoshell_moss.channels.ground_channel import new_ground_channel
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.ground import GroundSet

from ._ego_memento import EgoMementoManager
from .memento_channel import build_memento_channel

__all__ = ["build_dolores_channel"]


def build_dolores_channel(
    *,
    groundset: GroundSet,
    workspace_root: str | Path,
    memento_manager: EgoMementoManager | None = None,
    memento_root: str | Path | None = None,
    commit_anchor: Callable[[str], str | None] | None = None,
    frame_root: str | Path | None = None,
    init_frame: Frame | None = None,
    name: str = "ghost",
    description: str | None = None,
) -> MutableChannel:
    """组装 Dolores 的 ghost 根 channel.

    :param groundset: ghost 持有的 GroundSet — 注入给 ground 子 channel 共享, 不新建
        (两个 set 压同一目录 = 两个 snapshot owner).
    :param workspace_root: 子 channel 相对路径解析基点 (ghost home).
    :param memento_manager: ghost 持有的 memento 服务 — memento 子 channel 读它. None
        = 不挂记忆器官 (memento 未启用时).
    :param memento_root: 轨迹的磁盘位置 (进自解释). None = 不复述路径.
    :param commit_anchor: 主动落锚的后端 (memento 子 channel 的写面透传; 收 note 返回坐标,
        没有新东西可封时返回 None). None = 不挂 ``commit`` 命令.
    :param frame_root: 思维框架 (frame) 的发现根目录, 边界 = project home. None = 不挂
        frame 器官.
    :param init_frame: 开机注入的首帧 (来自 startup doc). None = frame channel 空启动,
        由模型自己 load / define. 需要 frame_root 一起提供才生效.
    :param name: channel 名 (runtime 以 ``ghost`` 为键注册, 同名保持一致).
    :param description: 覆盖默认描述.
    """
    if description is None:
        description = "Dolores' reflexive control surface — its sub-channels are its own organs."

    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        return (
            "## Ghost (reflexive)\n"
            "Your own control surface: the sub-channels are your organs, not external tools. "
            "Their state refreshes with your cognitive field; command signatures are reflected, "
            "not restated here."
        )

    chan.import_channels(
        new_ground_channel(
            groundset,
            workspace_root=workspace_root,
            render_root=False,
            name="ground",
            description=(
                "Your cognitive field (GROUND.md) — the law chain survives compaction; "
                "open/close mount a field as a sub-channel."
            ),
        )
    )
    if memento_manager is not None:
        chan.import_channels(
            build_memento_channel(
                memento_manager, storage_root=memento_root, commit_anchor=commit_anchor
            )
        )
    if frame_root is not None:
        chan.import_channels(
            new_frame_channel(
                root=frame_root,
                init_frame=init_frame,
                name="frame",
                description=(
                    "Your thinking frame — a question set you resolve from context "
                    "to hold a stable self-model across context loss."
                ),
            )
        )
    return chan
