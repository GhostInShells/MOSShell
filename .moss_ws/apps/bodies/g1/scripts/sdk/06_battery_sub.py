#!/usr/bin/env python3
"""
订阅 G1 电池/主板/机身IMU/里程计 — 被动感知 topic 真值清单。

修正记录 (2026-06-15):
  - 前任版本将 odommodestate 类型标为 IMUState_ — 错。odom 的 payload 是
    SportModeState_ (含 position/velocity/imu_state 聚合)。
  - 多 topic 候选名探测: 同时尝试 rt/<name> 和 rt/lf/<name>，找出 G1 真发哪个。
  - bms/mainboard/secondary_imu 已在前任 session (2026-06-15) 验证 rt/lf/<name>
    可读，保留为首选；rt/<name> 作为备选探测。

SDK 参考:
  unitree_sdk2py/idl/unitree_hg/msg/dds_/_BmsState_.py
  unitree_sdk2py/idl/unitree_hg/msg/dds_/_MainBoardState_.py
  unitree_sdk2py/idl/unitree_hg/msg/dds_/_IMUState_.py
  unitree_sdk2py/idl/unitree_go/msg/dds_/_SportModeState_.py  (odom payload)

用法: python 06_battery_sub.py <networkInterface>
"""
import sys
import time


def try_subscribe(topic, cls, label, timeout_ms=2000):
    """尝试订阅一个 topic，2s 超时。返回 True/False。"""
    from unitree_sdk2py.core.channel import ChannelSubscriber

    print(f"  订阅 {topic} ({label}, {cls.__name__})...")
    try:
        sub = ChannelSubscriber(topic, cls)
        sub.Init()
        msg = sub.Read(timeout=timeout_ms)
        if msg is None:
            print(f"    超时 — topic 不存在或类型不匹配")
            sub.Close()
            return False
        print(f"    OK ← 收到数据")
        # 摘要打印
        for f in dir(msg):
            if f.startswith('_'):
                continue
            val = getattr(msg, f)
            if isinstance(val, (int, float, str, bool)):
                print(f"      .{f} = {val}")
            elif hasattr(val, '__len__'):
                try:
                    n = len(val)
                    if n <= 12:
                        print(f"      .{f} = {list(val)}")
                    else:
                        print(f"      .{f} = <len={n}> {list(val)[:6]}...")
                except Exception:
                    pass
        sub.Close()
        return True
    except Exception as e:
        print(f"    FAIL — {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python 06_battery_sub.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        BmsState_, MainBoardState_, IMUState_,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    # (label, [(topic, cls), ...])
    # 每项尝试多个候选 topic，首个命中即停。
    targets = [
        ("电池 BmsState",      [("rt/lf/bmsstate", BmsState_),
                                ("rt/bmsstate", BmsState_)]),
        ("主板 MainBoardState", [("rt/lf/mainboardstate", MainBoardState_),
                                 ("rt/mainboardstate", MainBoardState_)]),
        ("机身 IMU (hg)",      [("rt/lf/secondary_imu", IMUState_),
                                ("rt/secondary_imu", IMUState_)]),
        ("里程计 SportModeState", [("rt/odommodestate", SportModeState_),
                                   ("rt/lf/odommodestate", SportModeState_),
                                   ("rt/sportmodestate", SportModeState_)]),
    ]

    summary = {}
    for label, candidates in targets:
        print(f"\n{'='*55}")
        print(f"目标: {label}")
        hit = None
        for topic, cls in candidates:
            if try_subscribe(topic, cls, label):
                hit = topic
                break
        summary[label] = hit

    print(f"\n{'='*55}")
    print("候选探测结果:")
    for label, hit in summary.items():
        if hit:
            print(f"  [OK]   {label:<28s} → {hit}")
        else:
            print(f"  [FAIL] {label:<28s} → 所有候选均超时")
    print("\n下一步: 把命中的 topic 名记入 docs/sdk-topics.md (订正前任清单)。")


if __name__ == "__main__":
    main()