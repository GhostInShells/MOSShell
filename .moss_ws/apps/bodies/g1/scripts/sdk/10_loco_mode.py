#!/usr/bin/env python3
"""
运动模式切换验证。高风险 — 涉及全身运动。
验证: LocoClient 模式切换 Damp → Sit → Start (非调试模式，内置运控兜底)。

SDK 参考:
  example/g1/high_level/g1_loco_client_example.py  — LocoClient 12 项交互菜单
  unitree_sdk2py/g1/loco/g1_loco_client.py          — LocoClient 实现
  src/unitree_sdk2_python/

安全:
  人类在场 + 机器人周围清空 (半径 2m 无障碍) + 遥控器握急停权 L2+B。
  本脚本仅做模式切换，不做移动。
  先确认 CheckMode 返回的当前模式。

前置:
  G1 开机 + 非调试模式 + 机器人周围清空 + 遥控器就绪
  source .venv/bin/activate
  python 00_import_verify.py

用法: python 10_loco_mode.py <networkInterface>
"""
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("用法: python 10_loco_mode.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    print("=" * 60)
    print("G1 运动模式切换验证")
    print("安全要求:")
    print("  1. 机器人周围 2m 内无任何障碍物或人")
    print("  2. 遥控器在手，L2+B 急停随时可用")
    print("  3. 本脚本仅做模式切换 (Damp→Sit→Start)")
    print("    不做移动。不做 ZeroTorque。")
    print("=" * 60)
    input("按 Enter 确认安全条件满足...")

    print(f"\n初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    # 先检查当前运动模式
    print("检查当前运动模式...")
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    code, result = msc.CheckMode()
    if code == 0:
        print(f"当前模式: {result}")
    else:
        print(f"WARN: CheckMode code={code}")

    # LocoClient 初始化
    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("LocoClient 初始化完成\n")

    # ── 模式切换序列 ──
    steps = [
        ("Damp",    "进入阻尼模式 (电机停转+阻尼)"),
        ("Sit",     "进入落座模式"),
        ("Start",   "进入主运控 (Start)"),
    ]

    for method, desc in steps:
        print(f"{'─'*40}")
        print(f"  {method}: {desc}")
        print(f"  3 秒后执行...")
        for i in range(3, 0, -1):
            print(f"    {i}...")
            time.sleep(1)

        fn = getattr(loco, method)
        code = fn()
        if code == 0:
            print(f"  OK: {method} 完成")
        else:
            print(f"  FAIL: {method} code={code}")
            print("  立即停止序列!")
            break

        time.sleep(2)

    print(f"\n{'='*40}")
    print("模式切换序列完成。")
    print("\n验证结论:")
    print("  [ ] Damp: 机器人是否进入阻尼？(LED 橙色)")
    print("  [ ] Sit:  机器人是否落座？(LED 绿色)")
    print("  [ ] Start: 机器人是否恢复主运控？")
    print("  [ ] 模式切换是否流畅？是否有异常抖动？")
    print("  [ ] SportModeState 的 fsm_id 是否跟随切换变化？")

if __name__ == "__main__":
    main()
