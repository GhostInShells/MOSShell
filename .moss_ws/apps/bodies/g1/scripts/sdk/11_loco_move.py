#!/usr/bin/env python3
"""
移动控制验证。最高风险 — 全身运动。
验证: 极低速 0.5s 定时前进 + 持续前进最短脉冲 + 慢速旋转

SDK 参考:
  example/g1/high_level/g1_loco_client_example.py  — Move 调用示例
  unitree_sdk2py/g1/loco/g1_loco_client.py          — Move(vx,vy,vyaw,continuous)
  src/unitree_sdk2_python/

安全:
  0.1 m/s 极慢速。定时移动仅 0.5~1.0 秒。
  需空旷场地 (前方 2m)。遥控器急停就绪。

前置:
  G1 开机 + 非调试模式 + 已执行 Start(站立运控) + 前方 2m 清空 + 遥控器急停
  source .venv/bin/activate
  python 00_import_verify.py

用法: python 11_loco_move.py <networkInterface>
"""
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("用法: python 11_loco_move.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print("=" * 60)
    print("G1 移动控制验证 — 极慢速 0.1 m/s | 短脉冲 | 遥控器急停")
    print("安全要求:")
    print("  1. G1 处于 Start(站立运控) 模式 — 落座模式 Move 无效")
    print("  2. 前方 2m 空旷无任何障碍")
    print("  3. 遥控器在手 | L2+B 随时急停")
    print("  4. 所有移动 0.5~1.0 秒内自动停止")
    print("=" * 60)
    input("按 Enter 确认安全条件满足...")

    print(f"\n初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("LocoClient 初始化完成\n")

    VX = 0.1   # 0.1 m/s

    # ── 测试 1: 极短定时前进 ──
    print("[测试 1] 极短定时前进 0.5 秒")
    print(f"  Move(vx={VX}, vy=0, vyaw=0)")
    print("  3 秒后执行...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)

    code = loco.Move(VX, 0, 0)  # 默认 duration=1s
    if code == 0:
        print("  OK: 移动指令已发送")
        time.sleep(0.5)
        code = loco.StopMove()
        print(f"  0.5s 后 StopMove: {'OK' if code == 0 else f'FAIL code={code}'}")
    else:
        print(f"  FAIL: Move code={code}")

    time.sleep(3)

    # ── 测试 2: 极慢速左右移动 ──
    print("\n[测试 2] 横移 1 秒 (vy=0.1)")
    print("  3 秒后执行...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)

    code = loco.Move(0, VX, 0)
    if code == 0:
        print("  OK: 横移中...")
        time.sleep(1.0)
        code = loco.StopMove()
        print(f"  1s 后 StopMove: {'OK' if code == 0 else f'FAIL code={code}'}")
    else:
        print(f"  FAIL: Move code={code}")

    time.sleep(3)

    # ── 测试 3: 慢速旋转 ──
    print("\n[测试 3] 慢速原地旋转 0.5 秒 (vyaw=0.2)")
    print("  3 秒后执行...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)

    code = loco.Move(0, 0, 0.2)
    if code == 0:
        print("  OK: 旋转中...")
        time.sleep(0.5)
        code = loco.StopMove()
        print(f"  0.5s 后 StopMove: {'OK' if code == 0 else f'FAIL code={code}'}")
    else:
        print(f"  FAIL: Move code={code}")

    time.sleep(3)

    # ── 测试 4: 持续前进最短脉冲 ──
    print("\n[测试 4] 持续前进 0.5 秒后手动停止")
    print(f"  Move(vx={VX}, vy=0, vyaw=0, continuous=True)")
    print("  3 秒后执行...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        time.sleep(1)

    code = loco.Move(VX, 0, 0, True)
    if code == 0:
        print("  持续模式移动中...")
        time.sleep(0.5)
        code = loco.StopMove()
        print(f"  0.5s 后 StopMove: {'OK' if code == 0 else f'FAIL code={code}'}")
    else:
        print(f"  FAIL: Move continuous code={code}")

    time.sleep(3)

    print(f"\n{'='*40}")
    print("移动测试完成。机器人应已停止。")
    print("\n验证结论:")
    print("  [ ] 定时 Move: 是否前进约 0.05m？")
    print("  [ ] 横移: 是否侧移约 0.1m？")
    print("  [ ] 旋转: 是否原地旋转约 5°？")
    print("  [ ] 持续 Move: StopMove 是否立即生效？")
    print("  [ ] L2+B 急停在所有阶段是否都能立即终止运动？")
    print("\n二阶实验 (后续，需更空旷场地):")
    print("  - 持续 Move 长距离 (2-3 米) 速度稳定性")
    print("  - 前进中切换方向 (vx→vy 不经过 StopMove)")

if __name__ == "__main__":
    main()
