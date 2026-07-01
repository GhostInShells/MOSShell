"""
Locomotion — G1 空间移动 channel (L4).

7 个 async 命令 (前后 / 左右横移 / 左右转 / 强停) 反射 runtime.locomotion, 通过 fsm
授权门控绑定物理安全边界.

runtime 依赖:
  - `ghoshell_moss_contrib.unitree.g1.runtime.locomotion` — 命令执行器
  - `ghoshell_moss_contrib.unitree.g1.runtime.story_202607_fsm` — 状态机授权

生命周期由集成层 (mode channels.py / app main.py) 在 sdk.bootstrap() 后
`locomotion.start()` + `fsm.start()`, channel 只用不启停.

授权分档 (每档需 FsmMode.WALK_RUN, AI 模式激活是根阀门):
- turn_left / turn_right / stop: L1+ (原地转身 + 强停, 低风险)
- walk_* / strafe_*: L2+ (空间位移, 需更高授权)

授权即时性:
- command 启动前 `available` 门控 shell 层调度. 不符合的命令 shell 不会派发.
- command 执行中 `fsm.warrant(guard)` 硬中断 — 授权中途撤销 (退 AI / 降 auth /
  掉出 WALK_RUN), warrant 转 ObserveError, 让 LLM 立即感知. runtime finally 走
  StopMove 兜底物理停止.

context_messages 每帧输出 <locomotion_authorization>: 当前 sport_mode + auth_level
+ 三组能力符合情况. LLM 通过它知道"我此刻能做什么 / 差什么条件".
物理感知连续采样 (motion FSM / imu 姿态) 归感知 channel 自身, 本 channel 不复述.

observe 分工:
- 移动命令 `-> str` return runtime 文本 (ReAct tool_result, 下次思考可见, 不打断).
- pre-check 失败 (整机不可用 / speed 非法) `raise_observe` — 那才是必须打断.
- warrant 硬中断 (授权撤销) 转 ObserveError, 属打断范畴 (与 pre-check 失败同类).
"""
from __future__ import annotations

import logging
from typing import Literal

from ghoshell_moss.core.blueprint.channel_builder import (
    new_channel, CommandUtil,
)
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.runtime import locomotion
from ghoshell_moss_contrib.unitree.g1.runtime import story_202607_fsm as fsm
from ghoshell_moss_contrib.unitree.g1.sdk import FsmMode
from ghoshell_moss_contrib.unitree.g1.channels._utils import (
    check_g1_available, check_channel_information,
)

logger = logging.getLogger("moss.g1.channels.locomotion")

__all__ = ["locomotion_channel"]


# ═════════════════════════════════════════════════════════════════════════════
# 授权谓词 — command.available + fsm.warrant guard 复用
# ═════════════════════════════════════════════════════════════════════════════

def _turn_available() -> bool:
    """L1+ (含) WALK_RUN. 转身 / 强停共用."""
    return fsm.need_fsm_state([([FsmMode.WALK_RUN], _TURN_AUTHS)])


def _move_available() -> bool:
    """L2+ (含) WALK_RUN. 前后 / 横移用."""
    return fsm.need_fsm_state([([FsmMode.WALK_RUN], _MOVE_AUTHS)])


# authorized levels — int lists.
# fsm 内部 `_auth_level: int = AUTH_LEVEL_MIN`, need_fsm_state 的 `in` 判定用 int
# 也成立. class AuthLevel 目前在 story_202607_fsm 里未定义 (被引用但缺 import/decl),
# 待 runtime 层补齐后可迁移到 `[fsm.AuthLevel.L1, ...]`.
# 值语义: 0 = 零权限, 1 = L1, 2 = L2, 3 = L3, 4 = L4 (fsm 常量 AUTH_LEVEL_MAX=3
# 但 _bump_auth_up 里封顶 L4, 保守列全 1..4).
_TURN_AUTHS: list[int] = [1, 2, 3, 4]
_MOVE_AUTHS: list[int] = [2, 3, 4]


# stop 授权与 turn 同档 (低风险). 显式 alias 让 command 装饰器语义清晰.
_stop_available = _turn_available


# ═════════════════════════════════════════════════════════════════════════════
# instruction — 通用契约, 不列命令名/授权细节 (授权情况由 context_messages 动态告知)
# ═════════════════════════════════════════════════════════════════════════════

_INSTRUCTION = """\
你可以让身体在空间中移动. 具体动作命令由本通道的命令面自解释.

坐标系 (相对机身): x 前 / -x 后, y 左 / -y 右, yaw 逆时针为正.

时间是一等公民 — 每个动作命令用 duration (秒) 控制持续时长.

授权与能力符合情况随时刷新在 <locomotion_authorization>, 依它判断当前能做什么.

安全:
- 你有重量与惯性, 未确认周围空间前不要发大 duration.
- 首次移动前用 0.5-1.0s 的小 duration 试探.
"""

locomotion_channel = new_channel(
    name="locomotion",
    description="G1 空间移动 — 前后走 / 左右横移 / 左右转 / 强停.",
)
locomotion_channel.build.instruction(_INSTRUCTION)


# ─── startup ─────────────────────────────────────────────────────────────────

@locomotion_channel.build.startup
async def _on_startup() -> None:
    locomotion.start()


# ═════════════════════════════════════════════════════════════════════════════
# context_messages — 每帧描述当前授权 + 三组能力符合情况
# ═════════════════════════════════════════════════════════════════════════════

@locomotion_channel.build.context_messages
async def _authorization_context() -> list[Message]:
    ai, sport, auth = fsm.read()
    if not ai:
        content = "AI 模式未激活, locomotion 全部不可用 (需先进入 AI 模式)"
    else:
        can_turn = _turn_available()
        can_move = _move_available()
        can_stop = _stop_available()
        lines = [f"FSM: {sport.name}, 授权: L{auth}"]
        lines.append(f"- 转身 (左右): {'可用' if can_turn else '不可用 (需 WALK_RUN + L1+)'}")
        lines.append(f"- 空间移动 (前后/横移): {'可用' if can_move else '不可用 (需 WALK_RUN + L2+)'}")
        lines.append(f"- 强停: {'可用' if can_stop else '不可用 (需 WALK_RUN + L1+)'}")
        content = "\n".join(lines)
    return [Message.new(tag="locomotion_authorization").with_content(content)]


# ═════════════════════════════════════════════════════════════════════════════
# 命令面
# ═════════════════════════════════════════════════════════════════════════════

_VALID_SPEEDS = ("low", "medium", "high")


def _check_speed(speed: str) -> None:
    if speed not in _VALID_SPEEDS:
        CommandUtil.raise_observe(
            f"speed 必须是 {'/'.join(_VALID_SPEEDS)}, 给的是 {speed}"
        )


@locomotion_channel.build.command(available=_move_available)
async def walk_forward(duration: float) -> str:
    """向前直行 (机身正面方向), 固定速度 0.25 m/s.

    duration: 持续时间 (秒), 必须 > 0.
    """
    check_g1_available()
    async with fsm.warrant(_move_available):
        return await locomotion.walk_forward(duration)


@locomotion_channel.build.command(available=_move_available)
async def walk_backward(duration: float) -> str:
    """向后直行 (机身正面方向的反方向), 固定速度 0.25 m/s.

    duration: 持续时间 (秒), 必须 > 0.
    """
    check_g1_available()
    async with fsm.warrant(_move_available):
        return await locomotion.walk_backward(duration)


@locomotion_channel.build.command(available=_move_available)
async def strafe_left(duration: float) -> str:
    """向左横移 (机身正面不变, 整体向左侧平移), 固定速度 0.25 m/s.

    duration: 持续时间 (秒), 必须 > 0.
    """
    check_g1_available()
    async with fsm.warrant(_move_available):
        return await locomotion.strafe_left(duration)


@locomotion_channel.build.command(available=_move_available)
async def strafe_right(duration: float) -> str:
    """向右横移, 固定速度 0.25 m/s.

    duration: 持续时间 (秒), 必须 > 0.
    """
    check_g1_available()
    async with fsm.warrant(_move_available):
        return await locomotion.strafe_right(duration)


@locomotion_channel.build.command(available=_turn_available)
async def turn_left(
    duration: float,
    speed: Literal["low", "medium", "high"] = "low",
) -> str:
    """原地左转 (yaw 增加方向). 通过 duration + speed 组合控制转角.

    粗略对照: low ≈ 17°/s, medium ≈ 34°/s, high ≈ 57°/s (实机标定中).

    duration: 持续时间 (秒), 必须 > 0.
    speed: "low" (谨慎打量) / "medium" (一般转身) / "high" (应急回头).
    """
    check_g1_available()
    _check_speed(speed)
    async with fsm.warrant(_turn_available):
        return await locomotion.turn_left(duration, speed)


@locomotion_channel.build.command(available=_turn_available)
async def turn_right(
    duration: float,
    speed: Literal["low", "medium", "high"] = "low",
) -> str:
    """原地右转 (yaw 减少方向).

    duration: 持续时间 (秒), 必须 > 0.
    speed: "low" (谨慎打量) / "medium" (一般转身) / "high" (应急回头).
    """
    check_g1_available()
    _check_speed(speed)
    async with fsm.warrant(_turn_available):
        return await locomotion.turn_right(duration, speed)


@locomotion_channel.build.command(blocking=False, available=_stop_available)
async def stop() -> str:
    """强制立即停止任何 active 移动. 不占命令队列, 可在其他移动命令期间插队执行.

    与移动命令之间的抢占不同 — stop 本身不启动新移动, 只把当前 active 打回 stand.
    """
    check_g1_available()
    # stop 是紧急插队, 不包 warrant — warrant 会跟自己制造的授权撤销事件互相干扰,
    # 且 stop 本身就是"授权撤销时的应急动作", 语义上不该自己再检查授权.
    return await locomotion.stop()


if __name__ == "__main__":
    # dev dump: monkey-patch fsm.read 遍历几个典型 scenario 的 prompt 输出.
    # 真实场景需 sdk.bootstrap + fsm.start, 不适合 dev dump.
    _scenarios: list[tuple[str, tuple[bool, FsmMode, int]]] = [
        ("ai_off",             (False, FsmMode.UNKNOWN,  0)),
        ("ai_on_stand_l0",     (True,  FsmMode.STAND,    0)),
        ("ai_on_walkrun_l0",   (True,  FsmMode.WALK_RUN, 0)),
        ("ai_on_walkrun_l1",   (True,  FsmMode.WALK_RUN, 1)),
        ("ai_on_walkrun_l2",   (True,  FsmMode.WALK_RUN, 2)),
    ]
    for _name, _state in _scenarios:
        fsm.read = lambda s=_state: s
        print(f"\n\n{'=' * 60}\nscenario: {_name} → {_state}\n{'=' * 60}")
        check_channel_information(locomotion_channel)
