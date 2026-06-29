#!/usr/bin/env python3
"""
26_arm_sdk_dds_joints_write — rt/arm_sdk 底层 DDS 写关节角的基础可行性

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

如果 25(录制能力探测) 失败 — G1 不暴露录制接口 — 那"录制+回放"和
"手臂轨迹动画"都必须走 rt/arm_sdk DDS 自己写关节角时间序列.

本脚本是这条路径的可行性验证:
  Q1. rt/arm_sdk publisher 在 Sport 模式下是否能写入 (与 LocoClient 内置控制共存)
  Q2. 关节角控制是否真的能让 G1 手臂动 (50Hz + 线性插值, 跟 example arm7 一样)
  Q3. 单只手腕(wrist_roll, 23-DoF 仅有的腕关节自由度)能否独立控制
  Q4. weight 字段 (motor_cmd[kNotUsedJoint=29].q) 控制使能, 0→1 渐进开启是否平稳

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + **Sport 模式** (mode_machine=6)
  2. 手臂 1m 半径无人无物
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate
  4. 遥控器在手

测试矩阵:
  阶段 1: 仅 weight 0→1 渐进 — 手臂不动, 但 G1 应当感知到外部接管
  阶段 2: 单关节小幅度运动 — LeftWristRoll 从当前角 → +30° → 当前角
  阶段 3: 双手对称小幅度 — 双肩 pitch 各 +15° → 复位
  阶段 4: weight 1→0 渐进释放, 内置控制接管

每阶段让你观察 + 打分: 是否平稳? 是否真的动了? 内置 Sport 平衡是否保持?

风险:
  这是底层 DDS 写入, 直接控制电机. kp=60 kd=1.5 是 example 默认, 偏硬.
  线性插值 3s 完成 30° 转动 = 平均 10°/s, 不算快但也不能轻视.
  任何异常 L2+B 急停 → 进 Damp → 我们的脚本 publisher 仍在写但 G1 不响应.

实测记录:
  2026-06-29 deepseek-v4-pro + 人类:
    weight 0→1: 分=2, 很快 (kp 偏硬). wrist +30°: 分=1, 平滑缓慢.
    双肩 +15°: 分=1, 向后运动. weight 1→0: 分=1, 小动作好.
    结论: rt/arm_sdk 底层关节控制可行. DDS publish 停 = 真中断.
    arm_trajectory channel 可做. kp/kd 需调软.
"""
import sys
import time
import threading
import math
from typing import Optional


PI = 3.141592654


# G1 23-DoF JointIndex (来自 example/g1/high_level/g1_arm7_sdk_dds_example.py)
class JointIdx:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    kNotUsedJoint = 29  # weight 控制


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def grade(label: str) -> tuple[int, str]:
    print()
    print(f"  {label} 打分:")
    print("    1 = 平稳, 真的动了, 符合预期")
    print("    2 = 完成但有抖动 / 路径不平滑")
    print("    3 = 完成但有明显异常 / 共存控制冲突")
    print("    4 = 失败 / 危险 / 关节不响应")
    while True:
        ans = prompt("输入 1-4")
        if ans in {'1', '2', '3', '4'}:
            note = prompt("简短补充")
            return int(ans), note
        print("  请输入 1-4")


def main():
    if len(sys.argv) < 2:
        print("用法: python 26_arm_sdk_dds_joints_write.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    print("=" * 70)
    print("26_arm_sdk_dds_joints_write — rt/arm_sdk 底层写关节角可行性")
    print("=" * 70)
    print()
    print("命题: 在 Sport 模式下, rt/arm_sdk 写 motor_cmd 是否真的能控制手臂?")
    print()
    print("4 阶段渐进:")
    print("  1. weight 0→1 渐进开启接管 (手臂不动, 但接管标志拉起)")
    print("  2. LeftWristRoll 单关节小幅度 (+30° 然后回原)")
    print("  3. 双肩 pitch 对称小幅度 (+15° 然后回原)")
    print("  4. weight 1→0 渐进释放, Sport 内置控制接管")
    print()
    print("⚠️ 这是底层 DDS 写入, 直接驱动电机. kp=60 kd=1.5 偏硬, 但运动幅度很小.")
    print("=" * 70)
    input("\n准备好了按 Enter 开始 >>> ")

    print(f"\n初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    # 订阅 LowState 拿当前关节角
    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    state_sub.Init()

    msg = state_sub.Read(timeout=2000)
    if msg is None:
        print("FAIL: LowState 不到")
        sys.exit(1)
    if msg.mode_machine != 6:
        print(f"!! 当前 fsm = {msg.mode_machine}, 不是 Sport(6)")
        prompt_continue("切到 Sport 后回车")
        msg = state_sub.Read(timeout=2000)
        if msg is None or msg.mode_machine != 6:
            print("仍不是 Sport, 退出.")
            sys.exit(1)
    print(f"OK: Sport")

    # 发布器
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    print("OK: arm_sdk publisher 就绪")

    # 关节 ID 列表 (我们要控制的一组)
    arm_joints = [
        JointIdx.LeftShoulderPitch, JointIdx.LeftShoulderRoll, JointIdx.LeftShoulderYaw,
        JointIdx.LeftElbow, JointIdx.LeftWristRoll,
        JointIdx.RightShoulderPitch, JointIdx.RightShoulderRoll, JointIdx.RightShoulderYaw,
        JointIdx.RightElbow, JointIdx.RightWristRoll,
    ]
    kp = 60.0
    kd = 1.5
    crc = CRC()

    def make_cmd(weight: float, target_q: dict[int, float], current_state) -> LowCmd_:
        """构造 LowCmd_ 消息. target_q 仅含明确指定的关节, 其他保持当前角度."""
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.motor_cmd[JointIdx.kNotUsedJoint].q = weight

        for j in arm_joints:
            cmd.motor_cmd[j].kp = kp
            cmd.motor_cmd[j].kd = kd
            cmd.motor_cmd[j].tau = 0.0
            cmd.motor_cmd[j].dq = 0.0
            if j in target_q:
                cmd.motor_cmd[j].q = target_q[j]
            else:
                cmd.motor_cmd[j].q = current_state.motor_state[j].q
        cmd.crc = crc.Crc(cmd)
        return cmd

    prompt_continue("最后确认手臂 1m 无物, 你持遥控器在 G1 后方")

    results = []

    # ── 阶段 1: weight 0→1 渐进 ──
    print("\n" + "=" * 70)
    print("阶段 1: weight 0 → 1, 渐进 3s (手臂保持当前姿态)")
    print("=" * 70)
    prompt_continue("回车开始")

    duration = 3.0
    rate_hz = 50
    interval = 1.0 / rate_hz
    t_start = time.monotonic()
    while time.monotonic() - t_start < duration:
        ratio = (time.monotonic() - t_start) / duration
        current = state_sub.Read(timeout=100)
        if current is None:
            continue
        cmd = make_cmd(weight=ratio, target_q={}, current_state=current)
        pub.Write(cmd)
        time.sleep(interval)
    print("  weight=1 已达, 接管标志已拉起")

    g1, n1 = grade("阶段 1 weight 渐进开启")
    results.append({'phase': '1. weight 0→1', 'grade': g1, 'note': n1})

    if g1 == 4:
        print("失败, 释放后退出.")
        for _ in range(50):
            cmd = make_cmd(0, {}, state_sub.Read(timeout=100) or msg)
            pub.Write(cmd); time.sleep(0.02)
        sys.exit(1)

    # ── 阶段 2: LeftWristRoll +30° → 复位 ──
    print("\n" + "=" * 70)
    print("阶段 2: LeftWristRoll +30° (3s) → 复位 (3s)")
    print("=" * 70)
    prompt_continue("回车开始")

    current = state_sub.Read(timeout=500)
    q_start = current.motor_state[JointIdx.LeftWristRoll].q
    q_target = q_start + 30.0 * PI / 180.0
    print(f"  起始角 = {q_start:.3f} rad, 目标角 = {q_target:.3f} rad")

    # 推进 3s
    t_start = time.monotonic()
    while time.monotonic() - t_start < 3.0:
        ratio = (time.monotonic() - t_start) / 3.0
        q_now = q_start * (1 - ratio) + q_target * ratio
        cur = state_sub.Read(timeout=100)
        if cur is None: continue
        cmd = make_cmd(1.0, {JointIdx.LeftWristRoll: q_now}, cur)
        pub.Write(cmd); time.sleep(interval)

    # 复位 3s
    t_start = time.monotonic()
    while time.monotonic() - t_start < 3.0:
        ratio = (time.monotonic() - t_start) / 3.0
        q_now = q_target * (1 - ratio) + q_start * ratio
        cur = state_sub.Read(timeout=100)
        if cur is None: continue
        cmd = make_cmd(1.0, {JointIdx.LeftWristRoll: q_now}, cur)
        pub.Write(cmd); time.sleep(interval)

    g2, n2 = grade("阶段 2 单关节运动")
    results.append({'phase': '2. wrist +30° / 复位', 'grade': g2, 'note': n2})

    # ── 阶段 3: 双肩 pitch +15° → 复位 ──
    print("\n" + "=" * 70)
    print("阶段 3: 双肩 ShoulderPitch +15° → 复位")
    print("=" * 70)
    prompt_continue("回车开始")

    current = state_sub.Read(timeout=500)
    l_start = current.motor_state[JointIdx.LeftShoulderPitch].q
    r_start = current.motor_state[JointIdx.RightShoulderPitch].q
    l_target = l_start + 15.0 * PI / 180.0
    r_target = r_start + 15.0 * PI / 180.0

    t_start = time.monotonic()
    while time.monotonic() - t_start < 3.0:
        ratio = (time.monotonic() - t_start) / 3.0
        l_now = l_start * (1 - ratio) + l_target * ratio
        r_now = r_start * (1 - ratio) + r_target * ratio
        cur = state_sub.Read(timeout=100)
        if cur is None: continue
        cmd = make_cmd(1.0, {JointIdx.LeftShoulderPitch: l_now, JointIdx.RightShoulderPitch: r_now}, cur)
        pub.Write(cmd); time.sleep(interval)

    t_start = time.monotonic()
    while time.monotonic() - t_start < 3.0:
        ratio = (time.monotonic() - t_start) / 3.0
        l_now = l_target * (1 - ratio) + l_start * ratio
        r_now = r_target * (1 - ratio) + r_start * ratio
        cur = state_sub.Read(timeout=100)
        if cur is None: continue
        cmd = make_cmd(1.0, {JointIdx.LeftShoulderPitch: l_now, JointIdx.RightShoulderPitch: r_now}, cur)
        pub.Write(cmd); time.sleep(interval)

    g3, n3 = grade("阶段 3 双肩对称运动")
    results.append({'phase': '3. 双肩 +15° / 复位', 'grade': g3, 'note': n3})

    # ── 阶段 4: weight 1→0 渐进释放 ──
    print("\n" + "=" * 70)
    print("阶段 4: weight 1 → 0, 渐进 3s (释放给 Sport 内置控制)")
    print("=" * 70)
    prompt_continue("回车开始")

    t_start = time.monotonic()
    while time.monotonic() - t_start < 3.0:
        ratio = 1.0 - (time.monotonic() - t_start) / 3.0
        cur = state_sub.Read(timeout=100)
        if cur is None: continue
        cmd = make_cmd(weight=ratio, target_q={}, current_state=cur)
        pub.Write(cmd); time.sleep(interval)

    g4, n4 = grade("阶段 4 weight 渐进释放")
    results.append({'phase': '4. weight 1→0', 'grade': g4, 'note': n4})

    state_sub.Close()

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("rt/arm_sdk 底层写入可行性汇总")
    print("=" * 70)
    for r in results:
        print(f"  {r['phase']:<25} 分={r['grade']}  {r['note']}")
    print()
    grades = [r['grade'] for r in results]
    if all(g == 1 for g in grades):
        verdict = "✓ rt/arm_sdk 可用 — 自造手臂轨迹动画的基础成立"
    elif max(grades) <= 2:
        verdict = "~ 可用但有抖动 — 调 kp/kd, 加更长 ramp"
    elif max(grades) <= 3:
        verdict = "! 跟 Sport 控制有冲突 — 需要更细的共存策略"
    else:
        verdict = "✗ 失败 — 自造轨迹动画路径不通, 需另想办法"
    print(f"  {verdict}")
    print()
    print("反馈给模型: 模型据此决定 arm_trajectory channel 是否可做.")


if __name__ == "__main__":
    main()
