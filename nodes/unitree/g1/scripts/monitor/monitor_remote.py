#!/usr/bin/env python3
"""
monitor_remote — 监控遥控器按键和摇杆信号

订阅 rt/lowstate, 解析 wireless_remote[40], 实时打印按键边沿和摇杆轴变化.

用法:
  python monitor_remote.py <networkInterface>
  python monitor_remote.py eth0

输出:
  每行 = [时间] key X ↓ down / ↑ up
        [时间] axis Lx active +0.523
"""

import sys
import time
import struct
import threading


KEY_BITS_DATA1 = [
    ('R1', 0), ('L1', 1), ('Start', 2), ('Select', 3),
    ('R2', 4), ('L2', 5), ('F1', 6), ('F3', 7),
]

KEY_BITS_DATA2 = [
    ('A', 0), ('B', 1), ('X', 2), ('Y', 3),
    ('Up', 4), ('Right', 5), ('Down', 6), ('Left', 7),
]


def parse_keys(data: bytes) -> dict[str, int]:
    d1, d2 = data[2], data[3]
    keys = {}
    for name, bit in KEY_BITS_DATA1:
        keys[name] = (d1 >> bit) & 1
    for name, bit in KEY_BITS_DATA2:
        keys[name] = (d2 >> bit) & 1
    return keys


def parse_axes(data: bytes) -> dict[str, float]:
    return {
        'Lx': struct.unpack('<f', data[4:8])[0],
        'Rx': struct.unpack('<f', data[8:12])[0],
        'Ry': struct.unpack('<f', data[12:16])[0],
        'Ly': struct.unpack('<f', data[20:24])[0],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python monitor_remote.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    print(f"初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    print("订阅 rt/lowstate 就绪")
    print()

    last_keys = {}
    last_axes = {}
    running = True

    def _poll():
        nonlocal last_keys, last_axes
        while running:
            msg = sub.Read(timeout=500)
            if msg is None:
                continue
            data = bytes(msg.wireless_remote)
            keys = parse_keys(data)
            axes = parse_axes(data)

            ts = time.strftime('%H:%M:%S')

            if last_keys:
                for name, val in keys.items():
                    if last_keys.get(name, 0) != val:
                        edge = '↓ down' if val else '↑ up'
                        print(f"  [{ts}] key {name:<6} {edge}")

            if last_axes:
                AXIS_DEAD = 0.15
                for name, val in axes.items():
                    prev = last_axes.get(name, 0.0)
                    if abs(val) > AXIS_DEAD and abs(prev) <= AXIS_DEAD:
                        print(f"  [{ts}] axis {name:<3} active {val:+.3f}")

            last_keys = keys
            last_axes = axes

    _thread = threading.Thread(target=_poll, daemon=True)
    _thread.start()

    print("按 Ctrl+C 退出")
    print()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        running = False
        _thread.join(timeout=2)
        sub.Close()


if __name__ == "__main__":
    main()
