#!/usr/bin/env python3
"""
21_arm_action_interruption — Action A 中途发 Action B(非 99) 的物理行为

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

设计 channel 时, arm 命令在 CTML 队列中是顺序执行的(channel 内 FIFO). 模型可能
连续编排多条 arm 命令: face wave → clap → hug. 如果模型紧凑地编排, 第 N 条命令
可能在第 N-1 条物理还没播完时就到达 G1.

SDK 文档 + example 没说这种情况下 G1 怎么处理. 三种可能:
  A. 覆盖: B 立刻接管, A 立刻停 — 跟 ExecuteAction(99) 一样的中断语义
  B. 排队: B 等 A 物理完成再播
  C. 拒绝: B 返回非零 code, A 继续播

每种语义对 channel 的实现影响完全不同:
  A → arm command 是天然可"插队"的, 中断 fallback 容易
  B → 必须在 channel 层做 await 等物理完成, 否则模型编排会被 G1 内部排队
  C → channel 必须自己跟踪当前是否在播放, 拒绝并发提交

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + **Sport 模式**(mode_machine=6) — arm action 必须 Sport
  2. 手臂 1m 半径内无人无物
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate
  4. 遥控器在手

测试矩阵:
  3 组测试, 每组都是 "触发 A → 等 X 秒 → 触发 B(非 99)"

  Test 1: hands up(15, 长) 0.5s 后发 clap(17, 短)
          — 大幅度动作中途打断成另一个不同动作

  Test 2: face wave(25, 单臂) 1.0s 后发 high wave(26, 双臂)
          — 单臂动作中途切到双臂动作

  Test 3: clap(17, 双臂在前) 0.5s 后发 hands up(15, 双臂上举)
          — 短动作早期切到长动作

  每次让你观察 B 触发后 G1 的反应:
    1 = 立刻替换(覆盖语义)
    2 = 等待 A 完成才动(排队语义)
    3 = 拒绝(B RPC code != 0, A 继续)
    4 = 异常 / 不可预测

  注意: B 触发后我会等 5s 让你观察 + 可能再让 G1 完成动作.
        中间 RPC code 也会输出.

风险:
  arm 动作有破坏性. 周围 1m 半径内绝对不能有任何物体.
  任何异常 L2+B.
"""
import sys
import time
import threading
from typing import Optional


# (A_id, A_name, A_estimated_dur, interrupt_delay, B_id, B_name, label)
TEST_CASES = [
    (15, "hands up", 4.0, 0.5, 17, "clap",      "Test 1: hands up 早期 → clap"),
    (25, "face wave", 3.0, 1.0, 26, "high wave", "Test 2: face wave 中期 → high wave"),
    (17, "clap",      3.0, 0.5, 15, "hands up",  "Test 3: clap 早期 → hands up"),
]


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def grade_behavior() -> tuple[int, str]:
    print()
    print("  B 触发后 G1 反应是:")
    print("    1 = 立刻替换 A, 开始播放 B (覆盖)")
    print("    2 = A 继续播完, 然后才播 B (排队)")
    print("    3 = B 被拒绝(看上面 RPC code), A 继续播 (拒绝)")
    print("    4 = 异常 / 行为不可预测")
    while True:
        ans = prompt("输入 1-4")
        if ans in {'1', '2', '3', '4'}:
            note = prompt("简短补充(直接回车跳过)")
            return int(ans), note
        print("  请输入 1-4")


def report_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("Action A → B (非 99) 中断行为汇总")
    print("=" * 70)
    print()
    print(f"{'测试':<48} {'A.code':<8} {'B.code':<8} {'分':<4} 备注")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<48} {r['a_code']:<8} {r['b_code']:<8} {r['grade']:<4} {r['note']}")
    print()

    grades = [r['grade'] for r in results]
    print("结论判定:")
    if all(g == 1 for g in grades):
        verdict = "✓ 一致覆盖语义 — arm command 天然可中断, channel 实现简单"
    elif all(g == 2 for g in grades):
        verdict = "~ 一致排队语义 — channel 必须等物理完成, await 时长 = action 真实播放时长"
    elif all(g == 3 for g in grades):
        verdict = "! 一致拒绝语义 — channel 必须自己跟踪 'in_progress', 防止并发提交"
    else:
        verdict = "? 不一致 — 行为依赖具体 action 组合, 需要更细的策略"
    print(f"  {verdict}")
    print()
    print("把汇总反馈给模型实例, 更新 design 文档的 'arm command 并发语义' 部分.")


def main():
    if len(sys.argv) < 2:
        print("用法: python 21_arm_action_interruption.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    print("=" * 70)
    print("21_arm_action_interruption — Action 中途切换的物理行为")
    print("=" * 70)
    print()
    print("命题: A 播放中发 B(B != 99), G1 是 覆盖/排队/拒绝?")
    print()
    print("3 组测试, 每组让你观察 + 打分.")
    print()
    print("安全: Sport 模式, 周围 1m 无物.")
    print("=" * 70)
    input("\n准备好了按 Enter 开始 >>> ")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()

    # 确认 Sport
    print("\n确认 Sport...")
    msg = sub.Read(timeout=2000)
    if msg is None:
        print("FAIL: LowState 不到. 退出.")
        sys.exit(1)
    if msg.mode_machine != 6:
        print(f"!! 当前 fsm = {msg.mode_machine}, 不是 Sport(6)")
        prompt_continue("切到 Sport 后回车")
        msg = sub.Read(timeout=2000)
        if msg is None or msg.mode_machine != 6:
            print("仍不是 Sport, 退出.")
            sys.exit(1)
    print("OK")

    prompt_continue("最后确认: 手臂 1m 半径内无人无物, 你在 G1 后方持遥控器")

    results = []

    for (a_id, a_name, a_dur, delay, b_id, b_name, label) in TEST_CASES:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        print(f"流程: ExecuteAction({a_id}) [{a_name}] → 等 {delay}s → ExecuteAction({b_id}) [{b_name}]")

        prompt_continue("准备好了回车")

        t0 = time.monotonic()
        a_code = arm.ExecuteAction(a_id)
        print(f"  [{time.monotonic()-t0:.2f}s] A 触发: code = {a_code}")
        if a_code != 0:
            print(f"  !! A 失败, 跳过本测试")
            continue

        time.sleep(delay)

        t_b = time.monotonic()
        b_code = arm.ExecuteAction(b_id)
        print(f"  [{t_b-t0:.2f}s] B 触发: code = {b_code}")

        # 观察期 5s, 然后等 A 或 B 自然结束
        print(f"  观察 5s...")
        time.sleep(5.0)

        # 收尾: 不管行为如何, 发 release 让 G1 回中
        print(f"  发 release(99) 收尾...")
        arm.ExecuteAction(99)
        time.sleep(3)

        grade, note = grade_behavior()
        results.append({
            'label': label,
            'a_id': a_id, 'a_code': a_code,
            'b_id': b_id, 'b_code': b_code,
            'grade': grade, 'note': note,
        })

        if grade == 4:
            print("\n!!! 异常 — 终止后续测试.")
            break

        prompt_continue("等 G1 静止 + 你准备好下一个测试")

    sub.Close()
    report_summary(results)


if __name__ == "__main__":
    main()
