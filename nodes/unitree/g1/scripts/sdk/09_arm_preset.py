#!/usr/bin/env python3
"""
手臂预设动作验证。低风险 — 坐姿下只动手臂。
验证: 基础动作 + 中断复位 (二阶实验)。

SDK 参考:
  example/g1/high_level/g1_arm_action_example.py  — 完整 16 个动作交互
  unitree_sdk2py/g1/arm/g1_arm_action_client.py    — G1ArmActionClient + action_map
  src/unitree_sdk2_python/

安全:
  G1 处于落座(Sit)模式。手臂周围无遮挡。
  人类在场 + 遥控器握急停权。

前置:
  G1 开机 + 落座模式 + 手臂周围无遮挡
  source .venv/bin/activate
  python 00_import_verify.py

用法: python 09_arm_preset.py <networkInterface>
"""
import sys
import time

ACTION_MAP = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "heart": 20,
    "right heart": 21,
    "reject": 22,
    "right hand up": 23,
    "x-ray": 24,
    "face wave": 25,
    "high wave": 26,
    "shake hand": 27,
}

def main():
    if len(sys.argv) < 2:
        print("用法: python 09_arm_preset.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    print("=" * 60)
    print("G1 手臂预设动作验证")
    print("安全: G1 落座模式 | 手臂周围无遮挡 | 遥控器急停就绪")
    print("=" * 60)
    input("按 Enter 继续...")

    print(f"\n初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()
    print("G1ArmActionClient 初始化完成\n")

    print("可用动作:")
    for name, aid in ACTION_MAP.items():
        print(f"  {aid:3d}  {name}")
    print()

    # ── 测试 1: 基础动作 + 自动复位 ──
    action_name = "face wave"
    action_id = ACTION_MAP[action_name]

    print(f"[测试 1] 基础动作: {action_name} (id={action_id})")
    input("按 Enter 执行...")

    print("  3 秒后执行...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)

    code = arm.ExecuteAction(action_id)
    print(f"  ExecuteAction: {'OK' if code == 0 else f'FAIL code={code}'}")

    time.sleep(3)
    print("  复位...")
    code = arm.ExecuteAction(ACTION_MAP["release arm"])
    print(f"  release arm: {'OK' if code == 0 else f'FAIL code={code}'}")

    time.sleep(3)

    # ── 测试 2: 中断复位 ──
    print(f"\n[测试 2] 中断复位: 执行长动作 → 中途打断 → 温柔复位")
    long_action = "hands up"
    print(f"  动作: {long_action} (id={ACTION_MAP[long_action]})")
    input("按 Enter 执行...")

    print("  3 秒后执行...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)

    code = arm.ExecuteAction(ACTION_MAP[long_action])
    print(f"  ExecuteAction: {'OK' if code == 0 else f'FAIL code={code}'}")

    print("  等待 1.5 秒后打断...")
    time.sleep(1.5)
    print("  发送 release arm (中途打断)...")
    code = arm.ExecuteAction(ACTION_MAP["release arm"])
    print(f"  release arm: {'OK' if code == 0 else f'FAIL code={code}'}")

    time.sleep(3)

    # ── 测试 3: 序列动作 ──
    print(f"\n[测试 3] 动作序列: wave → release → clap → release")
    input("按 Enter 执行...")

    for name in ["face wave", "clap"]:
        print(f"\n  3 秒后: {name}...")
        for i in range(3, 0, -1):
            print(f"    {i}...")
            time.sleep(1)

        code = arm.ExecuteAction(ACTION_MAP[name])
        print(f"  {name}: {'OK' if code == 0 else f'FAIL code={code}'}")
        time.sleep(3)

        print("  复位...")
        code = arm.ExecuteAction(ACTION_MAP["release arm"])
        print(f"  release: {'OK' if code == 0 else f'FAIL code={code}'}")

        time.sleep(3)

    print("\n验证结论:")
    print("  [ ] 基础动作是否流畅完成？")
    print("  [ ] 中断复位: release arm 是否成功打断并温柔复位？")
    print("  [ ] 动作序列: 连续两个不同动作是否正常？")
    print("  [ ] 坐姿下手臂动作是否影响身体平衡？")
    print("\n二阶实验 (后续):")
    print("  - ExecuteAction 是阻塞还是非阻塞？ (源码注释: 阻塞)")
    print("  - 能否在动作执行期间发送新动作？ (队列 or 覆盖 or 拒绝)")

if __name__ == "__main__":
    main()
