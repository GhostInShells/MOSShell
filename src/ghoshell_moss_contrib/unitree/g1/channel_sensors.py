"""
G1 感知 channel — sensors 父 channel + 各 sub-sensor.

═══════════════════════════════════════════════════════════════════════════════
⚠️ 2026-06-29 校正标记 — 当前实装方向偏差, 不要照抄
═══════════════════════════════════════════════════════════════════════════════

人类工程师指出: 模型(我)还没理解 sensors 的核心范式 — context_messages vs command.

原始设计意图(见 design/2026-06-28_channel_architecture.md "感知统一进 context_messages"):
  - 所有感知数据通过 channel.build.context_messages() 进入模型上下文
    (滑动窗口, 过期即忘, 模型每帧都看到当前 N 条)
  - pop() 命令的作用是把"当前在 context_messages 里的快照"作为 command result 返回
    → 这一动作把数据从"过期即忘的上下文"提升到"持久 memory"
  - 模型主动决定"什么值得记住", pop 是这个决定的执行接口

本文件当前实装错在哪:
  - 把 pop() 写成了"现场读 state.py 返回字典" — 只是把 state 套层壳, 不是 sensor 范式
  - 完全没用 channel.build.context_messages() — 上下文这一层根本没接入
  - "window/open/close" 的概念在已实装 sensor (motion/remote/.../joints) 里完全没体现, 但这些
    才是 sensor 形态的本质 — sensor 是"持续观察的窗口", 不是"按需查询的 RPC"

正确形态应该是:
  - 每个 sensor 在 build.context_messages() 里返回最近 N 条 (从环形缓冲读)
  - 子线程 / running hook / monitor 把数据持续填入环形
  - pop() 才是"现场快照 → 进 memory" 的桥, 而不是 sensor 的全部
  - 开关/window 参数控制环形大小 + context 是否暴露

本期为什么先不改:
  - 上下文 + 滑动窗口 + context_messages 接入是新一波结构, 校正成本 > 收益
  - matrix 重构进行中, MCP 体验流程不通, 模型(我)没法自己写一遍跑一遍找感觉
  - 留着错的形态当反例, 比硬改更值钱 — 下一波清楚要做什么

下一波要做什么:
  1. 单独读一份 sensors 范式的"对的例子" — 翻 ghoshell_moss 已有 channel 看 build.context_messages 用法
  2. 设计每个 sensor 的"持续观察源" — 高频感知(motion/remote)直接读 state.py 还有意义,
     低频汇总(trajectory/odometry)需要环形缓冲 + monitor 采样
  3. pop() 退化为"快照→memory"的桥, 不再是 sensor 的主入口

本文件其余部分保留, 给下一波做反例参考.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from . import state


# ═══════════════════════════════════════════════════════════════════════════════
# 已实装 sensor — 直接读 state.py
# ═══════════════════════════════════════════════════════════════════════════════


def _build_motion_sensor() -> MutableChannel:
    chan = new_channel(name="motion", description="G1 FSM 模式 + tick")

    @chan.build.command(always_observe=True)
    async def pop() -> dict:
        """返回当前 MotionState 字段."""
        m = state.motion()
        return {"fsm_mode": m.fsm_mode, "tick": m.tick}

    return chan


def _build_remote_sensor() -> MutableChannel:
    chan = new_channel(name="remote", description="G1 遥控器实时状态")

    @chan.build.command(always_observe=True)
    async def pop() -> dict:
        """返回当前 RemoteState — 摇杆 + 按键 + is_estop."""
        r = state.remote()
        return {
            "lx": r.lx, "ly": r.ly, "rx": r.rx, "ry": r.ry,
            "buttons": {
                "a": r.a, "b": r.b, "x": r.x, "y": r.y,
                "up": r.up, "down": r.down, "left": r.left, "right": r.right,
                "l1": r.l1, "l2": r.l2, "r1": r.r1, "r2": r.r2,
                "start": r.start, "select": r.select, "f1": r.f1, "f3": r.f3,
            },
            "is_estop": r.is_estop,
        }

    return chan


def _build_battery_sensor() -> MutableChannel:
    chan = new_channel(name="battery", description="G1 电池状态 (SOC/电压/温度)")

    @chan.build.command(always_observe=True)
    async def pop() -> dict:
        b = state.battery()
        return {
            "soc": b.soc,
            "soh": b.soh,
            "voltage": b.voltage,
            "current": b.current,
            "cycle": b.cycle,
            "temperature_max": max(b.temperature) if b.temperature else None,
        }

    return chan


def _build_imu_sensor() -> MutableChannel:
    chan = new_channel(name="imu", description="G1 机身 IMU (rpy/gyro/accel/quat)")

    @chan.build.command(always_observe=True)
    async def pop() -> dict:
        i = state.imu()
        return {
            "rpy": i.rpy,
            "gyro": i.gyro,
            "accel": i.accel,
            "quat": i.quat,
        }

    return chan


def _build_health_sensor() -> MutableChannel:
    chan = new_channel(name="health", description="G1 主板温度 + 风扇")

    @chan.build.command(always_observe=True)
    async def pop() -> dict:
        h = state.health()
        return {
            "board_temp": h.board_temp,
            "fan_state": h.fan_state,
            "voltages": h.voltages,
        }

    return chan


def _build_joints_sensor() -> MutableChannel:
    """关节快照. 第一版默认全 23-DoF, 不做"挑选展示"过滤(后续加)."""
    chan = new_channel(name="joints", description="G1 全身关节 (q/dq/tau/mode)")

    # 23-DoF 关节命名 (来自 SDK G1JointIndex)
    JOINT_NAMES = [
        "L_HipPitch", "L_HipRoll", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
        "R_HipPitch", "R_HipRoll", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
        "WaistYaw", "WaistRoll", "WaistPitch",
        "L_ShldPitch", "L_ShldRoll", "L_ShldYaw", "L_Elbow", "L_WristRoll", "L_WristPitch", "L_WristYaw",
        "R_ShldPitch", "R_ShldRoll", "R_ShldYaw", "R_Elbow", "R_WristRoll", "R_WristPitch", "R_WristYaw",
    ]

    @chan.build.command(always_observe=True)
    async def pop() -> dict:
        """返回 23-DoF 关节角度/速度/力矩."""
        js = state.joints()
        out: dict[str, Any] = {}
        for i, name in enumerate(JOINT_NAMES):
            if i >= len(js.joints):
                break
            j = js.joints[i]
            out[name] = {"q": round(j.q, 4), "dq": round(j.dq, 4), "tau": round(j.tau, 3), "mode": j.mode}
        return out

    return chan


# ═══════════════════════════════════════════════════════════════════════════════
# 占位 sensor — 等实验或下一波实装
# ═══════════════════════════════════════════════════════════════════════════════


def _build_trajectory_sensor() -> MutableChannel:
    """空间移动轨迹. 1Hz 关键帧 × N 环形.

    TODO: 实装环形缓冲. 需要在 _monitor 加一个 1Hz 采样任务,
    把 motion + imu + sport_mode_state 写入环形.
    """
    chan = new_channel(name="trajectory", description="G1 移动轨迹关键帧 (未实装)")

    @chan.build.command()
    async def open(window: int = 30) -> str:
        raise NotImplementedError("trajectory sensor: not implemented yet (TODO: monitor 1Hz sampler)")

    @chan.build.command()
    async def close() -> str:
        raise NotImplementedError("trajectory sensor: not implemented yet")

    @chan.build.command()
    async def pop() -> dict:
        raise NotImplementedError("trajectory sensor: not implemented yet")

    return chan


def _build_odometry_sensor() -> MutableChannel:
    """里程计轨迹. 类似 trajectory, 数据源不同.

    TODO: 实装. 需要订阅 rt/odommodestate (SportModeState_ 类型,
    2026-06-16 实测确认 topic 存在).
    """
    chan = new_channel(name="odometry", description="G1 里程计关键帧 (未实装)")

    @chan.build.command()
    async def open(window: int = 30) -> str:
        raise NotImplementedError("odometry sensor: not implemented yet (TODO: subscribe rt/odommodestate)")

    @chan.build.command()
    async def close() -> str:
        raise NotImplementedError("odometry sensor: not implemented yet")

    @chan.build.command()
    async def pop() -> dict:
        raise NotImplementedError("odometry sensor: not implemented yet")

    return chan


def _build_actions_sensor() -> MutableChannel:
    """被动执行 action 历史. passenger 模式下 G1 自主行为的记录.

    TODO: 实装. 需要 channel_arm 在每次 ExecuteAction 时记一笔,
    或订阅 rt/arm/action/state (22 实验后定).
    """
    chan = new_channel(name="actions", description="G1 最近执行 action 历史 (未实装)")

    @chan.build.command()
    async def pop(limit: int = 10) -> list:
        raise NotImplementedError("actions sensor: not implemented yet (TODO: depends on 22 experiment)")

    return chan


def _build_asr_sensor() -> MutableChannel:
    """G1 内置 ASR sensor.

    TODO: 等 23 实验给出 _Call(1002, ...) 的调用约定 + 结果获取路径(同步返回 vs DDS topic).
    本期完全占位, 任何命令调用都 raise NotImplementedError.
    """
    chan = new_channel(name="asr", description="G1 ASR 最近识别结果 (未实装, 等 23 实验)")

    @chan.build.command()
    async def open(window: int = 5, language: str = "zh") -> str:
        raise NotImplementedError(
            "asr sensor: not implemented yet. "
            "blocked by sdk/23_asr_api_probe.py — need to determine _Call(1002, ...) "
            "convention and whether results arrive via RPC or DDS topic."
        )

    @chan.build.command()
    async def close() -> str:
        raise NotImplementedError("asr sensor: blocked by sdk/23")

    @chan.build.command()
    async def pop() -> list:
        raise NotImplementedError("asr sensor: blocked by sdk/23")

    return chan


# ═══════════════════════════════════════════════════════════════════════════════
# Sensors 父 channel
# ═══════════════════════════════════════════════════════════════════════════════


def build_sensors_channel() -> MutableChannel:
    """构建 sensors 父 channel, 含所有 sub-sensor.

    已实装: motion / remote / battery / imu / health / joints
    占位 (NotImplementedError): trajectory / odometry / actions / asr
    本期不做: vision (脸部摄像头物理设备未集成)

    所有 sensor 都需要 bootstrap 完成 — pop 内部读 state.py 会 raise 如果未启动.
    """
    sensors = new_channel(
        name="sensors",
        description="G1 感知子树. 每个 sensor 提供 pop() 把当前快照拉进 memory.",
    )

    sensors.import_channels(
        _build_motion_sensor(),
        _build_remote_sensor(),
        _build_battery_sensor(),
        _build_imu_sensor(),
        _build_health_sensor(),
        _build_joints_sensor(),
        _build_trajectory_sensor(),
        _build_odometry_sensor(),
        _build_actions_sensor(),
        _build_asr_sensor(),
    )

    sensors.build.instruction(
        "G1 感知 channel. 每个 sub-sensor 有 pop() 读当前快照. "
        "未实装项调用会 raise NotImplementedError."
    )

    return sensors
