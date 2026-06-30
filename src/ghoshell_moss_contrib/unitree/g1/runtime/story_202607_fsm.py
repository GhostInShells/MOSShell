"""
用户故事 2026-07 交付版本 — 语义常量.

两套常量:
  L1: G1 物理 FSM 模式 — 来源 LocoClient.GetFsmId() / rt/sportmodestate.
  L2: MOSS 授权等级 — F1 逐级向上, F3 归零.

这些只是名字和值的映射. 怎么组合、怎么随着事件变迁,
由 channel available() 和其他编排层决定, 不在此模块表达.
"""

from __future__ import annotations

import enum


# ═══════════════════════════════════════════════════════════════════════════════
# L1: G1 物理 FSM 模式
# ═══════════════════════════════════════════════════════════════════════════════

class FsmMode(int, enum.Enum):
    """G1 FSM 模式 ID.

    来源: LocoClient.GetFsmId() / rt/sportmodestate.
    mode_machine (LowState 字段) 是 DoF 配置字节, 不是 FSM.
    """

    UNKNOWN = -1 # 启动前的状态, 没有得到正确的信号同步.
    DAMP = 0
    ZERO_TORQUE = 0  # 遥控器语音"零力矩" — 与 DAMP 同 ID, 物理语义不同.
    SIT = 3  # 落座 — 开机默认, 安全姿态.
    STAND = 4  # 锁定站立.
    START = 5  # 基础站立 (预备).
    SPORT = 6  # 运控全开.

    # 待实测:
    #   DANCE — R1+B 舞蹈运控, FSM ID 未确认.
    #   DEBUG — L2+R2 调试模式, FSM ID 未确认.


# ═══════════════════════════════════════════════════════════════════════════════
# L2: MOSS 授权等级
# ═══════════════════════════════════════════════════════════════════════════════

class AuthLevel(enum.IntEnum):
    """授权等级 — 递增.

    F1 逐级向上授权, F3 归零到当前 FSM 支持的基础等级.
    站起是单向的 — Ghost 不能自己坐下.
    """

    SENSORS = 0  # 感知 + led + speaker + system. 开机后始终可用.
    ARMS = 1  # + 手臂 keyframe animation. 站起后默认获得.
    MOVE = 2  # + 运动控制 (限速). Sport 模式下 F1 授权获得.


# ═══════════════════════════════════════════════════════════════════════════════
# 按键名 — 状态机关注的子集, 与 g1.sdk._buttons.VALID_BUTTONS 对齐
# ═══════════════════════════════════════════════════════════════════════════════

BTN_F1 = "f1"  # Ghost trigger — 输入信号 + 向上授权.
BTN_F3 = "f3"  # Ghost interrupt — 归零.
BTN_START = "start"  # Channel interrupt — 中断当前 command loop.
BTN_L2 = "l2"  # 与 B 组合 = 硬件急停 Damp, 不可绕过.
BTN_B = "b"
BTN_L1 = "l1"  # 与组合键切换机体大状态机 — 人不经过 Ghost.
