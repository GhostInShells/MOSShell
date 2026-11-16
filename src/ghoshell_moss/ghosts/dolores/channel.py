"""Dolores 的 ghost 根 channel — 反身性控制面 | 集成 | alpha

根节点在 Dolores 自身的环境里组装: 闭包持有 ghost 的资源 (memento manager, ghost
home), 把子 channel 挂成静态子节点. 由 ``Ghost.channel()`` 返回, runtime 以 ``ghost``
名注册, CTML 路径即 ``ghost.*``.

子 channel 都是 ghost 自己的器官 —— ``ghost.memento`` 是它的记忆, ``ghost.model`` 是它
的大脑. **认知场不在器官之列**: ghost home 的场内容由 ``_runtime`` 渲染进 instruction,
直接作为认知本身 (见 ghost-ground-governance FEATURE).

Example:
    from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

    chan = build_dolores_channel(memento_manager=mm)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ghoshell_moss.channels.ground_channel import new_ground_channel
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from ._ego_memento import EgoMementoManager
from .memento_channel import build_memento_channel
from .mindset.prompter import PromptManager
from .model_channel import ModelOrgan, identity_text, new_model_channel

__all__ = ["build_dolores_channel"]


def build_dolores_channel(
    *,
    project_ground: str | Path | None = None,
    memento_manager: EgoMementoManager | None = None,
    prompt_manager: PromptManager | None = None,
    commit_anchor: Callable[[str], str] | None = None,
    checkout_hook: Callable[[str, str, str], str] | None = None,
    model: ModelOrgan | None = None,
    name: str = "ghost",
    description: str | None = None,
) -> MutableChannel:
    """组装 Dolores 的 ghost 根 channel.

    :param project_ground: 项目根的场目录. 给了就挂 ``ghost.project_ground``; None = 项目根
        没有 GROUND.md, 不挂. 边界就是项目根本身, 不是 ghost home.
    :param memento_manager: ghost 持有的 memento 服务 — memento 子 channel 读它. None
        = 不挂记忆器官 (memento 未启用时).
    :param prompt_manager: prompt registry — memento 子 channel 读 ``memento.note``
        的 active 正文作为 ``commit`` 的 note 规则.
    :param commit_anchor: 主动落锚的后端 (memento 子 channel 的写面透传; 收 note, 返回
        一条给模型的回执). None = 不挂 ``commit`` 命令.
    :param checkout_hook: 换线的后端 (收 name / title / note, 返回回执). None = 不挂
        ``checkout`` 命令 —— 转场归 ego, channel 只做校验与排队.
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
            "Your own control surface — the sub-channels are your organs, not external tools. "
            "Their state refreshes with your cognitive field."
        )

    if project_ground is not None:
        # 项目场是环境，不是器官: 用 channel 是因为要能随时拉一张项目面的渲染，
        # 而它不该占 ghost home 的那份 instruction 预算.
        chan.import_channels(
            new_ground_channel(
                project_ground,
                name="project_ground",
                description=(
                    "The project's cognitive field — call `render` for a face of any "
                    "position inside the project."
                ),
            )
        )
    if memento_manager is not None:
        chan.import_channels(
            build_memento_channel(
                memento_manager,
                prompt_manager=prompt_manager,
                commit_anchor=commit_anchor,
                checkout_hook=checkout_hook,
            )
        )
    if model is not None:
        chan.import_channels(new_model_channel(model))

        # 温数据面: 我实际在跑哪个模型、哪一档. named notice 的规矩 —— 变才重发, 没变零成本;
        # 来源是 request/header 的观测 (真在跑什么), 不是目录默认值 (新会话从哪开始).
        @chan.build.named_notices
        async def _model_identity() -> dict[str, str | None]:
            return {"model": identity_text(model.read_identity())}
    return chan
