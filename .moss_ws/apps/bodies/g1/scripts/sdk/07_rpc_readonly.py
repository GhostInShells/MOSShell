#!/usr/bin/env python3
"""
RPC 只读合集 — 四个无副作用 RPC 调用的返回验证。
验证: ServiceList, CheckMode, GetActionList, GetVolume 的返回格式和内容。

SDK 参考:
  unitree_sdk2py/b2/robot_state/robot_state_client.py     — RobotStateClient
  unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py — MotionSwitcherClient
  unitree_sdk2py/g1/arm/g1_arm_action_client.py           — G1ArmActionClient
  unitree_sdk2py/g1/audio/g1_audio_client.py              — AudioClient
  src/unitree_sdk2_python/

前置:
  G1 开机 + RPC 服务运行
  source .venv/bin/activate
  python 00_import_verify.py

用法: python 07_rpc_readonly.py <networkInterface>
"""
import sys
import json

def main():
    if len(sys.argv) < 2:
        print("用法: python 07_rpc_readonly.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.b2.robot_state.robot_state_client import RobotStateClient
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    all_ok = True

    # ── 1. RobotStateClient.ServiceList() — unitree_sdk2py/b2/robot_state/ ──
    print("=" * 50)
    print("1. RobotStateClient.ServiceList()")
    try:
        rsc = RobotStateClient()
        rsc.SetTimeout(5.0)
        rsc.Init()
        code, services = rsc.ServiceList()
        if code == 0 and services:
            print(f"OK: 发现 {len(services)} 个服务:")
            for s in services:
                print(f"  {s.name:<25s} status={s.status}  protect={s.protect}")
        else:
            print(f"FAIL: ServiceList code={code}, services={services}")
            all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    # ── 2. MotionSwitcherClient.CheckMode() — unitree_sdk2py/comm/motion_switcher/ ──
    print("\n2. MotionSwitcherClient.CheckMode()")
    try:
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        code, result = msc.CheckMode()
        if code == 0:
            print(f"OK: CheckMode result = {json.dumps(result)}")
            if result.get('name'):
                print(f"  当前运控模式: {result['name']}")
            else:
                print("  当前无运控模式运行 (调试模式?)")
        else:
            print(f"FAIL: CheckMode code={code}")
            all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    # ── 3. G1ArmActionClient.GetActionList() — unitree_sdk2py/g1/arm/ ──
    print("\n3. G1ArmActionClient.GetActionList()")
    try:
        arm = G1ArmActionClient()
        arm.SetTimeout(5.0)
        arm.Init()
        code, data = arm.GetActionList()
        if code == 0:
            if isinstance(data, list):
                print(f"OK: {len(data)} 个可用动作")
                for item in data:
                    name = item.get('name', item)
                    print(f"  {name}")
            else:
                print(f"OK (raw): {json.dumps(data)[:200]}")
        else:
            print(f"FAIL: GetActionList code={code}")
            all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    # ── 4. AudioClient.GetVolume() — unitree_sdk2py/g1/audio/ ──
    print("\n4. AudioClient.GetVolume()")
    try:
        audio = AudioClient()
        audio.SetTimeout(5.0)
        audio.Init()
        code, vol = audio.GetVolume()
        if code == 0:
            print(f"OK: Volume = {vol}")
        else:
            print(f"FAIL: GetVolume code={code}")
            all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    print(f"\n{'='*50}")
    if all_ok:
        print("全部 4 项 RPC 只读验证通过。")
    else:
        print("部分验证失败 — 检查 G1 是否开机、RPC 服务是否运行。")

if __name__ == "__main__":
    main()
