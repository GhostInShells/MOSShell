#!/usr/bin/env python3
"""
RPC 只读合集 — G1 RPC 服务存在性与返回格式验证 (无副作用)。

修正记录 (2026-06-15):
  - 前任版本把 RobotStateClient.ServiceList() 放第一个 + 5s 超时。RobotStateClient
    (b2.robot_state) 在所有 G1 examples 中从未出现，G1 bus 上很可能根本没有
    "robot_state" 服务。前任 session 报"import 失败"应是 Init/ServiceList 阻塞
    被误诊。本版本: 顺序后置 + 独立 try + 3s 超时 + 明确"G1 可能不可用"标注。
  - 顺序改为: MotionSwitcher → Arm → Audio → RobotState(可选)，确保确定可用
    的先跑出结论，可疑的放最后。

确认在 G1 examples 中出现的 RPC 客户端 (= 确定可用):
  unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py — MotionSwitcherClient ✓
  unitree_sdk2py/g1/arm/g1_arm_action_client.py                  — G1ArmActionClient ✓
  unitree_sdk2py/g1/audio/g1_audio_client.py                     — AudioClient ✓

未在 G1 examples 出现 (= 可能不可用):
  unitree_sdk2py/b2/robot_state/robot_state_client.py            — RobotStateClient ?

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

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    results = {}

    # ── 1. MotionSwitcherClient.CheckMode() — 确定可用 ──
    print("=" * 55)
    print("1. MotionSwitcherClient.CheckMode()  [G1 examples 确认]")
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(3.0)
        msc.Init()
        code, result = msc.CheckMode()
        if code == 0:
            print(f"  OK: CheckMode = {json.dumps(result)}")
            if isinstance(result, dict) and result.get('name'):
                print(f"  当前运控模式: {result['name']}")
            else:
                print("  当前无运控模式运行 (调试模式 / 未启动)")
            results['MotionSwitcher.CheckMode'] = True
        else:
            print(f"  FAIL: code={code}")
            results['MotionSwitcher.CheckMode'] = False
    except Exception as e:
        print(f"  FAIL: {e}")
        results['MotionSwitcher.CheckMode'] = False

    # ── 2. G1ArmActionClient.GetActionList() ──
    print("\n" + "=" * 55)
    print("2. G1ArmActionClient.GetActionList()  [G1 examples 确认]")
    try:
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
        arm = G1ArmActionClient()
        arm.SetTimeout(3.0)
        arm.Init()
        code, data = arm.GetActionList()
        if code == 0:
            if isinstance(data, list):
                print(f"  OK: {len(data)} 个动作 (源码 action_map 是 17 项)")
                for item in data[:20]:
                    name = item.get('name', item) if isinstance(item, dict) else item
                    print(f"    {name}")
            else:
                print(f"  OK (raw): {json.dumps(data)[:200]}")
            results['Arm.GetActionList'] = True
        else:
            print(f"  FAIL: code={code}")
            results['Arm.GetActionList'] = False
    except Exception as e:
        print(f"  FAIL: {e}")
        results['Arm.GetActionList'] = False

    # ── 3. AudioClient.GetVolume() ──
    print("\n" + "=" * 55)
    print("3. AudioClient.GetVolume()  [G1 examples 确认]")
    try:
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        audio = AudioClient()
        audio.SetTimeout(3.0)
        audio.Init()
        code, vol = audio.GetVolume()
        if code == 0:
            # 注意: GetVolume 返回 (code, dict) 不是 (code, int)
            print(f"  OK: Volume = {vol}  (注意返回是 dict)")
            results['Audio.GetVolume'] = True
        else:
            print(f"  FAIL: code={code}")
            results['Audio.GetVolume'] = False
    except Exception as e:
        print(f"  FAIL: {e}")
        results['Audio.GetVolume'] = False

    # ── 4. RobotStateClient.ServiceList() — G1 上可能不可用 ──
    print("\n" + "=" * 55)
    print("4. RobotStateClient.ServiceList()  [G1 examples 未引用 — 可能 G1 不可用]")
    print("   设 3s 超时；如阻塞或失败属于预期。")
    try:
        from unitree_sdk2py.b2.robot_state.robot_state_client import RobotStateClient
        rsc = RobotStateClient()
        rsc.SetTimeout(3.0)
        rsc.Init()
        code, services = rsc.ServiceList()
        if code == 0 and services:
            print(f"  OK: 发现 {len(services)} 个服务:")
            for s in services:
                print(f"    {s.name:<25s} status={s.status}  protect={s.protect}")
            results['RobotState.ServiceList'] = True
        else:
            print(f"  FAIL/未响应: code={code}, services={services}")
            print(f"  → 结论候选: G1 RPC bus 上不存在 'robot_state' 服务")
            results['RobotState.ServiceList'] = False
    except ImportError as e:
        print(f"  IMPORT FAIL (SDK 路径变更?): {e}")
        results['RobotState.ServiceList'] = False
    except Exception as e:
        print(f"  FAIL: {e}")
        results['RobotState.ServiceList'] = False

    print(f"\n{'='*55}")
    print("RPC 只读验证结果:")
    for name, ok in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}")

    print("\n下一步:")
    print("  - 全 OK → 4 个 RPC 接口都可用，进入 08 音频灯光实验")
    print("  - RobotState FAIL 是预期 → 确认结论，从可用 RPC 清单中删除")
    print("  - MotionSwitcher/Arm/Audio FAIL → 检查 G1 业务进程是否运行")


if __name__ == "__main__":
    main()