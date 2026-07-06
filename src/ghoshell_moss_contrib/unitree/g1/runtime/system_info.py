"""
System Info runtime — G1 电池 + 主板状态打包读.

不同于 motion / imu / arm_joints (有自己的 daemon + ring buffer), system_info 没有
独立 daemon. 它的数据源 (rt/lf/bmsstate, rt/lf/mainboardstate) 是**低频** DDS topic,
由 sdk._monitor 已经在订阅维护. 这里只做"按需打包读取"的薄壳:

  - 一个 read() 函数返回当前 SystemInfoSnapshot.
  - helper to_xml_text / to_message 给 channel 包装成 command 返回值.
  - 无 start / stop / drain / listener. 它就是一个 stateless query.

设计动机: 电池 / 主板信息对模型是"偶尔关心"的状态 (问"我电量多少"或"温度高不高"),
不需要持续推上下文也不需要采样轨迹. channel 把它暴露成一个 query command,
模型按需调.

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import system_info

    bootstrap(nic="eth0")
    snap = system_info.read()
    print(system_info.to_xml_text(snap))
"""
from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1 import sdk

logger = logging.getLogger("moss.g1.runtime.system_info")


# ── 数据契约 ──────────────────────────────────────────────────────────────

class SystemInfoSnapshot(BaseModel):
    """G1 电池 + 主板系统状态的一次打包快照. 一次 read 取一份."""

    # 电池
    battery_soc: int = Field(
        ...,
        description="电量百分比 [0, 100]. 关键决策字段 — 低于 20 应提示充电.",
    )
    battery_soh: int = Field(
        ...,
        description="电池健康度 [0, 100]. 反映电池本身老化程度, 通常较稳定.",
    )
    battery_voltage: float = Field(
        ...,
        description="电池总电压 (V).",
    )
    battery_current: float = Field(
        ...,
        description="电池电流 (A). 负值 = 放电中, 正值 = 充电中.",
    )
    battery_temperature_max: int = Field(
        ...,
        description="所有电芯温度的最大值 (°C). 异常高温 (>50°C) 警告.",
    )
    battery_cycle: int = Field(
        ...,
        description="充放电循环次数. 用于评估电池剩余寿命.",
    )

    # 主板
    board_temp: int = Field(
        ...,
        description="主板温度 (°C). 超过 70°C 警告.",
    )
    fan_running: bool = Field(
        ...,
        description="是否有任何风扇正在转动 (转速 > 0). 高温时应为 True.",
    )

    # 数据健康度
    last_update_seconds_ago: float = Field(
        ...,
        description="距离 sdk 收到上一帧 LowState 过了多少秒. "
                    "正常 < 0.01s; 超过 1s 表示 DDS 链路可能有问题.",
    )
    captured_at: float = Field(
        default_factory=time.time,
        description="本次 read 的本地时间.",
    )
    source: str = Field(default="g1.system", description="数据来源固定常量.")


# ── 公开接口 ─────────────────────────────────────────────────────────────

def read() -> SystemInfoSnapshot:
    """一次打包读取所有系统信息.

    前置: sdk.bootstrap() 已完成. 电池/主板 topic 是低频, bootstrap 后可能要
    等几秒才到首帧 — 未到则 raise.

    Raises:
        RuntimeError: sdk monitor 未启动, 或 battery / mainboard 首帧未到.
    """
    b = sdk.battery()
    h = sdk.health()
    last = sdk.last_update()

    return SystemInfoSnapshot(
        battery_soc=b.soc,
        battery_soh=b.soh,
        battery_voltage=round(b.voltage, 2),
        battery_current=round(b.current, 2),
        battery_temperature_max=max(b.temperature) if b.temperature else 0,
        battery_cycle=b.cycle,
        board_temp=h.board_temp,
        fan_running=any(s > 0 for s in h.fan_state),
        last_update_seconds_ago=round(max(0.0, time.monotonic() - last), 3),
    )


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────

def to_xml_text(s: SystemInfoSnapshot) -> str:
    """SystemInfoSnapshot → XML. channel 包装成 query command 返回值."""
    discharging = s.battery_current < 0
    current_arrow = "↓" if discharging else "↑"
    return (
        f'<{s.source} ts="{s.captured_at:.3f}">\n'
        f'  battery: soc={s.battery_soc}% soh={s.battery_soh}% '
        f'voltage={s.battery_voltage:.2f}V '
        f'current={s.battery_current:+.2f}A{current_arrow} '
        f'cell_temp_max={s.battery_temperature_max}°C cycles={s.battery_cycle}\n'
        f'  mainboard: temp={s.board_temp}°C '
        f'fans={"running" if s.fan_running else "off"}\n'
        f'  link: last_update={s.last_update_seconds_ago:.3f}s ago\n'
        f'</{s.source}>'
    )


def to_message(s: SystemInfoSnapshot) -> Message:
    return Message.new(
        tag=s.source,
        attributes={
            "soc": s.battery_soc,
            "board_temp": s.board_temp,
        },
        timestamp=True,
    ).with_content(to_xml_text(s))
