#!/usr/bin/env python3
"""
订阅 G1 电池和主板状态。
验证: BmsState_ 和 MainBoardState_ 类型路径正确，字段可读。

SDK 参考:
  docs/sdk-topics.md   — topic→type 映射
  src/unitree_sdk2_python/unitree_sdk2py/idl/
  注: SDK 无对应 example，需用 ChannelSubscriber 裸订阅

前置: 同 04_lowstate_sub.py
用法: python 06_battery_sub.py <networkInterface>
"""
import sys
import time

def try_subscribe(topic, cls, label):
    """尝试订阅一个 topic，打印接收到的数据"""
    from unitree_sdk2py.core.channel import ChannelSubscriber

    print(f"\n{'='*50}")
    print(f"订阅 {topic} ({label})...")
    try:
        sub = ChannelSubscriber(topic, cls)
        sub.Init()
        msg = sub.Read(timeout=5000)
        if msg is not None:
            print(f"OK: {topic} 数据:")
            for f in dir(msg):
                if not f.startswith('_'):
                    val = getattr(msg, f)
                    if isinstance(val, (int, float, str)):
                        print(f"  .{f} = {val}")
                    elif hasattr(val, '__len__') and len(val) < 20:
                        print(f"  .{f} = {list(val)[:10]}")
            sub.Close()
            return True
        else:
            print(f"WARN: {topic} 超时 — topic 可能不存在或类型不匹配")
            sub.Close()
            return False
    except Exception as e:
        print(f"FAIL: {topic} — {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python 06_battery_sub.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    results = {}

    # rt/lf/bmsstate — 电池
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_
        results['bms'] = try_subscribe("rt/lf/bmsstate", BmsState_, "电池")
    except ImportError:
        print("WARN: BmsState_ 不可 import — 尝试从 default 模块")
        try:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__BmsState_ as BmsState_
            results['bms'] = try_subscribe("rt/lf/bmsstate", BmsState_, "电池 (default)")
        except ImportError:
            print("FAIL: BmsState_ 无法导入")
            results['bms'] = False

    # rt/lf/mainboardstate — 主板
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MainBoardState_
        results['mainboard'] = try_subscribe("rt/lf/mainboardstate", MainBoardState_, "主板")
    except ImportError:
        print("WARN: MainBoardState_ 不可 import — 尝试 default 模块")
        try:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__MainBoardState_ as MainBoardState_
            results['mainboard'] = try_subscribe("rt/lf/mainboardstate", MainBoardState_, "主板 (default)")
        except ImportError:
            print("FAIL: MainBoardState_ 无法导入")
            results['mainboard'] = False

    # rt/lf/secondary_imu — 机身 IMU
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_
        results['sec_imu'] = try_subscribe("rt/lf/secondary_imu", IMUState_, "机身IMU")
    except ImportError:
        print("WARN: IMUState_ (hg) 不可 import")
        results['sec_imu'] = False

    # rt/lf/odommodestate — 里程计 (go2 类型)
    try:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_ as GoIMUState_
        results['odom'] = try_subscribe("rt/lf/odommodestate", GoIMUState_, "里程计")
    except ImportError:
        print("WARN: IMUState_ (go2) 不可 import")
        results['odom'] = False

    print(f"\n{'='*50}")
    print("验证结论:")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL/未确认"
        print(f"  [{status}] {name}")
    print("\n对照 docs/sdk-topics.md 类型验证清单。")
    print("FAIL 的类型可能需要从 default 模块导入或 topic 名不对。")

if __name__ == "__main__":
    main()
