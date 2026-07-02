"""
Arms channel — G1 手臂 L2 命名调用 channel (L4).

4 个 async 命令反射 runtime.arms 的命名动作面, 通过 fsm 授权门控绑定物理安全边界.

runtime 依赖:
  - `ghoshell_moss_contrib.unitree.g1.runtime.arms` — L2 命令执行器
  - `ghoshell_moss_contrib.unitree.g1.runtime.story_202607_fsm` — 状态机授权

生命周期由集成层 (mode channels.py) 在 sdk.bootstrap() 后 arms.start() +
fsm.start(), channel 只用不启停.

授权 (L2+ WALK_RUN 或 STAND):
  - arm 动作幅度大, 需要机身稳定 (Sport 模式)
  - L2+ 才开放 (跟 walk/strafe 同档 — 空间位移与手臂操作视为同等风险)
  - 中断三基础缺失 (碰撞反馈 / 复位可靠性 / 首帧过渡), 只做 L2 命名调用

observe 分工:
  - 动作命令 -> str: return runtime 文本 (ReAct tool_result, 下次思考可见, 不打断)
  - pre-check 失败 (整机不可用) raise_observe — 那才是必须打断
  - warrant 硬中断 (授权撤销中途) 转 ObserveError

设计参考:
  - locomotion channel (同 pattern) — L1+ 转身 / L2+ 移动分档
  - FEATURE.md "arms 能力金字塔" — 本模块处于 L2 层
"""
from __future__ import annotations

import logging

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.runtime import arms
from ghoshell_moss_contrib.unitree.g1.runtime import story_202607_fsm as fsm
from ghoshell_moss_contrib.unitree.g1.sdk import FsmMode
from ghoshell_moss_contrib.unitree.g1.channels._utils import (
    check_g1_available, check_channel_information,
)

logger = logging.getLogger("moss.g1.channels.arms")

__all__ = ["arms_channel"]


# ═════════════════════════════════════════════════════════════════════════════
# 授权谓词 — L2+ 手臂动作, 需 STAND 或 WALK_RUN (机身稳定)
# ═════════════════════════════════════════════════════════════════════════════

_ARMS_AUTHS: list[int] = [2, 3, 4]


def _arms_available() -> bool:
    """L2+ (含) STAND / WALK_RUN. 所有 arm 动作共用."""
    return fsm.need_fsm_state([
        ([FsmMode.STAND, FsmMode.WALK_RUN], _ARMS_AUTHS),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# instruction — 通用契约, 具体命令签名自解释
# ═════════════════════════════════════════════════════════════════════════════

_INSTRUCTION = """\
你的双臂可以做命名动作 (挥手 / 鼓掌 / 比心 / 复位).

授权模型:
- 每个命令的可用性根据当前 FSM + 授权动态刷新, 见 <arms_authorization>.
- 需 L2+ 授权 + FSM 站立 (STAND) 或 走跑 (WALK_RUN).

时序特点:
- arm 动作是 G1 主板 ExecuteAction, **不可中断** — 一个动作发出后必须等它跑完.
- 命令返回时手臂已自动复位到自然位 (release_arm 内建).
- arm 忙时新命令直接被拒绝 (Observe 明说 "arm busy"), 不打断当前动作.

安全:
- 手臂动作幅度大, 未确认周围空间前不要做. 首次执行前告知在场的人 "我要挥手 / 鼓掌".
- 若不确定动作是否完成, 主动调用 release 复位.
"""

arms_channel = new_channel(
    name="arms",
    description="G1 手臂命名动作 (挥手 / 鼓掌 / 比心 / 复位). 不可中断, 忙时拒绝.",
)
arms_channel.build.instruction(_INSTRUCTION)


# ─── startup ─────────────────────────────────────────────────────────────────

@arms_channel.build.startup
async def _on_startup() -> None:
    arms.start()


# ═════════════════════════════════════════════════════════════════════════════
# context_messages — 每帧描述当前授权 + 忙碌状态
# ═════════════════════════════════════════════════════════════════════════════

@arms_channel.build.context_messages
async def _authorization_context() -> list[Message]:
    ai, sport, auth = fsm.read()
    if not ai:
        content = "AI 模式未激活, arms 全部不可用 (需先进入 AI 模式)"
    else:
        available = _arms_available()
        current = arms.current_action()
        lines = [f"FSM: {sport.name}, 授权: L{auth}"]
        lines.append(f"- 手臂动作: {'可用' if available else '不可用 (需 STAND/WALK_RUN + L2+)'}")
        if current is not None:
            lines.append(f"- 当前忙: {current} 正在执行, 新命令会被拒绝")
        else:
            lines.append("- 当前空闲")
        content = "\n".join(lines)
    return [Message.new(tag="arms_authorization").with_content(content)]


# ═════════════════════════════════════════════════════════════════════════════
# 命令面
# ═════════════════════════════════════════════════════════════════════════════

@arms_channel.build.command(available=_arms_available)
async def wave() -> str:
    """挥手打招呼. 大幅高位挥手, ~4s 自动复位.

    return: Observe 文本 ("<observe>wave completed after Ns</observe>").
    """
    check_g1_available()
    return await arms.wave()


@arms_channel.build.command(available=_arms_available)
async def clap() -> str:
    """鼓掌. ~3s 自动复位."""
    check_g1_available()
    return await arms.clap()


@arms_channel.build.command(available=_arms_available)
async def heart() -> str:
    """双手比心. ~4s 自动复位."""
    check_g1_available()
    return await arms.heart()


@arms_channel.build.command(available=_arms_available)
async def release() -> str:
    """手动温柔复位. 空闲时主动"收手"回自然位.

    动作跑完自动 release, 这个命令让模型可以主动收手 (e.g. 不确定当前状态时).
    """
    check_g1_available()
    return await arms.release()


if __name__ == "__main__":
    check_channel_information(arms_channel)
