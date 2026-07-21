"""
_system_info_sen_read — System info 一次打包读取 + helper 输出.

场景:
  system_info 没有 daemon, 是一个 stateless query. 这个脚本只做一件事:
  bootstrap 后调一次 read(), 打印 SystemInfoSnapshot + helper to_xml_text.

  这是 channel 真实使用 scenario 的最小模拟:
    "模型问 '我电量多少' / '主板温度高吗', channel 调 system_info.read()
     + to_xml_text 包装成 command 返回值."

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._system_info_sen_read <nic>

前置:
  - G1 已开机
  - bmsstate / mainboardstate 都是低频 topic, bootstrap 后可能要 1-3s 才到
    首帧. 脚本最多重试 10s; 仍未收到则 raise 退出.

预期:
  [snapshot] SystemInfoSnapshot(
    battery_soc=85, battery_soh=98, battery_voltage=...,
    battery_current=-1.20, battery_temperature_max=28,
    battery_cycle=42, board_temp=45, fan_running=False,
    last_update_seconds_ago=0.003)
  [xml]
  <g1.system ts="...">
    battery: soc=85% soh=98% voltage=29.40V current=-1.20A↓ ...
    mainboard: temp=45°C fans=off
    link: last_update=0.003s ago
  </g1.system>
"""
from __future__ import annotations

import sys
import time

from ghoshell_moss_contrib.unitree.g1.runtime import system_info
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


def main(nic: str) -> int:
    print(f"[1/2] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)

    print("[2/2] system_info.read() (重试至 battery/mainboard 首帧到达, 最多 10s) ...")
    deadline = time.time() + 10.0
    snap = None
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            snap = system_info.read()
            break
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    if snap is None:
        print(f"[failed] system_info.read() 持续失败: {last_err}")
        return 1

    print()
    print("=" * 64)
    print("[snapshot]")
    print(snap.model_dump_json(indent=2))
    print()
    print("[xml]")
    print(system_info.to_xml_text(snap))
    print("=" * 64)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
