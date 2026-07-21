"""
G1 FSM 模式 ID 常量. 来源: LocoClient._Call(7001) RPC.
官方文档 ID 表 — 2026-07-01 实机验证 (Damp=1, Stand=4, WalkRun=801).
"""
import enum


class FsmMode(int, enum.Enum):
    UNKNOWN = -1
    ZERO_TORQUE = 0     # 零力矩 — 电机无阻尼
    DAMP = 1            # 阻尼 — 电机有阻尼, 可进预备
    SQUAT_POS = 2       # 位控下蹲
    SIT = 3             # 位控落座 — 开机默认
    STAND = 4           # 锁定站立 (预备模式)
    REGULAR = 500       # 常规运控
    REGULAR_3DOF = 501  # 常规运控-3Dof-waist
    LIE_STAND = 702     # 躺起
    BALANCE_SQUAT = 706 # 平衡下蹲/蹲起
    WALK_RUN = 801      # 走跑运控 29dof (8.6.x.x+ 更新为 802)
