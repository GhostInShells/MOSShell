#!/usr/bin/env python3
"""
18_arm_release_behavior — 验证 ExecuteAction(99) "release arm" 的物理行为

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本(给实机执行人/未来实例)
═══════════════════════════════════════════════════════════════════════════════

整个 channel 体系把"危险命令的 fallback"作为 warrant 事务的核心. arm 类命令的 fallback
是 ExecuteAction(99) = "release arm". 但 SDK 和文档都没说这个动作的物理行为是什么.

两种可能:
  A. 缓慢插值回中性姿态(挂在身侧) — 这是 fallback 想要的
  B. 撤销手臂控制权, 关节落回默认 — 可能突然垂落, 硬动作

只有 A 才能把 arm 命令包进 warrant. B 的话, arm 一旦发起就不可中断, channel 设计要重做.

如果观察到的是 B 或者更糟(脱力), 整个 arm 类的 warrant 方案废弃, 改走:
  - 要么 arm 命令一旦发起就不允许中断, 等播完
  - 要么走 rt/arm_sdk 底层 DDS 自己写插值复位 — 工程量大, 但才有真正的"缓慢复位"

═══════════════════════════════════════════════════════════════════════════════
执行人指引 — 你不需要动脑, 按步骤做即可
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + Sport 模式(站立运控) — 因为 arm 动作要求 Sport
  2. G1 周围空旷, 手臂活动半径内无人/无物
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate
  4. 遥控器在手, 任何异常按 L2+B 急停

测试矩阵:
  本脚本会跑 3 次 release 测试, 每次先触发一个有显著姿态变化的动作, 然后立刻调
  ExecuteAction(99), 你观察 release 是缓慢插值还是突然变化.

  测试 1: hands up (id=15) → release  — 手举得很高, release 时 "落下" 行为很明显
  测试 2: clap (id=17) → release        — 手在身体正前方, release 路径更短
  测试 3: face wave (id=25) → release   — 单臂在脸侧, 不对称释放对比

  每次测试中间会暂停, 你可以让 G1 完全静止 + 报告观察, 再继续下一次.

观察重点(对每次 release):
  Q1. release 是"插值平滑下降"还是"突然下降"?
  Q2. release 用时大约多少秒?
  Q3. 关节有没有"脱力下落"的感觉(垂直、加速)?
  Q4. 是否有可闻的电机减速声?

记录方法:
  脚本会引导你打分: 缓慢插值 / 中速过渡 / 突然变化 / 脱力(危险).
  你按数字 1/2/3/4 回答, 脚本汇总.

风险:
  arm 动作本身有破坏性. 周围 1m 半径内不能有任何物体或人. 任何时刻可 L2+B 急停.
"""
import sys
import time
import threading

ARM_ACTIONS_TO_TEST = [
    # (action_id, action_name, description, estimated_duration_sec)
    (15, "hands up",    "两手举高, 释放路径最长, 最容易看出差别", 4.0),
    (17, "clap",        "拍手, 双手在身体前方", 3.0),
    (25, "face wave",   "单手在脸侧挥, 不对称释放", 3.0),
]

RELEASE_ID = 99  # "release arm" 来自 SDK action_map


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ")


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def grade_release_behavior() -> tuple[int, str]:
    """提示执行人对刚才观察到的 release 打分."""
    print()
    print("  对刚才的 release 行为打分:")
    print("    1 = 缓慢插值平滑下降(理想 fallback)")
    print("    2 = 中速过渡, 能看出在控制, 但有点快")
    print("    3 = 突然变化, 但仍受控(不是自由落体)")
    print("    4 = 脱力 / 自由落体感(危险, 不能用)")
    while True:
        ans = prompt("输入 1-4")
        if ans.strip() in {'1', '2', '3', '4'}:
            note = prompt("简短补充观察(如'有电机声/平滑/抖动', 没补充直接回车)").strip()
            return int(ans), note
        print("  请输入 1-4")


def report_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("ExecuteAction(99) 行为观测汇总")
    print("=" * 70)
    print()
    print(f"{'测试':<20} {'打分':<6} {'用时':<8} {'备注'}")
    print("-" * 70)
    for r in results:
        print(f"{r['action_name']:<20} {r['grade']:<6} {r['observed_duration']:<8} {r['note']}")
    print()

    grades = [r['grade'] for r in results]
    avg = sum(grades) / len(grades) if grades else 0

    print("结论判定:")
    if all(g == 1 for g in grades):
        verdict = "✓ 一致缓慢插值 — release 可信, 可作为 arm warrant 的 fallback"
    elif max(grades) <= 2:
        verdict = "~ 缓慢到中速 — 大概率可用, 但需要看具体 action 是否一致, 在 channel 层加个保守等待"
    elif max(grades) <= 3:
        verdict = "! 部分动作 release 偏快 — 需评估对每个 action 单独的 fallback 时长"
    else:
        verdict = "✗ 有脱力/危险行为 — arm warrant 方案废弃, 改走 rt/arm_sdk 自造插值"

    print(f"  {verdict}")
    print(f"  平均打分: {avg:.1f}")
    print()
    print("把这份汇总反馈给模型实例, 更新 design/2026-06-28_channel_architecture.md 的 'arm fallback' 部分.")


def main():
    if len(sys.argv) < 2:
        print("用法: python 18_arm_release_behavior.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    print("=" * 70)
    print("18_arm_release_behavior — ExecuteAction(99) 物理行为验证")
    print("=" * 70)
    print()
    print("命题: ExecuteAction(99) 是'缓慢插值复位'还是'瞬间脱控/突然变化'.")
    print()
    print("将测试 3 个 arm action × ExecuteAction(99) 序列, 每次让你打分.")
    print()
    print("安全:")
    print("  - 必须 Sport 模式")
    print("  - 周围 1m 半径内不能有任何物体或人")
    print("  - 任何异常按 L2+B 急停")
    print("=" * 70)
    input("\n准备好了按 Enter 开始 >>> ")

    # ── 初始化 ──
    print(f"\n初始化 DDS (domain=0, interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()
    print("OK: ArmClient 就绪")

    # ── 确认 Sport 模式 ──
    print("\n确认 G1 处于 Sport 模式...")
    msg = sub.Read(timeout=2000)
    if msg is None:
        print("FAIL: LowState 收不到. 检查 DDS / ufw / G1 状态.")
        sys.exit(1)
    fsm = msg.mode_machine
    if fsm != 6:
        print(f"!! 当前 fsm_mode = {fsm}, 不是 Sport(6).")
        print("   请用遥控器切到 Sport 模式后再继续(通常 L2+A 或类似组合).")
        prompt_continue("切到 Sport 后回车")
        msg = sub.Read(timeout=2000)
        if msg is None or msg.mode_machine != 6:
            print(f"   仍不是 Sport(实际 = {msg.mode_machine if msg else 'None'}). 退出.")
            sys.exit(1)
    print("OK: G1 处于 Sport 模式")

    prompt_continue("最后一次确认: G1 周围 1m 半径内无人无物")

    # ── 逐个测试 ──
    results = []

    for action_id, action_name, desc, est_dur in ARM_ACTIONS_TO_TEST:
        print("\n" + "=" * 70)
        print(f"测试: action_id={action_id}  名称={action_name}")
        print(f"说明: {desc}")
        print(f"预估动作时长: {est_dur}s (G1 内部播放)")
        print("=" * 70)

        prompt_continue("准备好了回车 — 然后我会触发动作")

        print(f"  -> ExecuteAction({action_id}) [{action_name}]")
        code = arm.ExecuteAction(action_id)
        if code != 0:
            print(f"  !! RPC 失败 code={code}, 跳过本测试")
            continue
        print("  RPC OK. 等待动作播完...")

        # 等动作播完 — 预估 + 1s 余量
        time.sleep(est_dur + 1.0)
        print(f"  动作应已完成. 接下来触发 release.")

        prompt_continue("再次确认周围安全, 然后回车触发 release")

        t_before = time.time()
        print(f"  -> ExecuteAction({RELEASE_ID}) [release arm]")
        code = arm.ExecuteAction(RELEASE_ID)
        t_after = time.time()
        rpc_latency = t_after - t_before
        if code != 0:
            print(f"  !! release RPC 失败 code={code}, 跳过本测试")
            continue
        print(f"  release RPC 返回 OK. RPC 用时 {rpc_latency*1000:.0f}ms")
        print(f"  现在观察 G1 手臂行为...")

        # 等 release 物理过程 — 给执行人 5s 观察
        time.sleep(5.0)

        observed_dur_str = prompt("release 用时大约多少秒(0.5/1/2/3/...) ?").strip()
        grade, note = grade_release_behavior()

        results.append({
            'action_id': action_id,
            'action_name': action_name,
            'rpc_latency_ms': int(rpc_latency * 1000),
            'observed_duration': observed_dur_str,
            'grade': grade,
            'note': note,
        })

        if grade == 4:
            print("\n  !!! 打分 4 = 脱力/危险. 立刻终止后续测试, 直接进入汇总.")
            break

        # 让 G1 完全静止再下一个
        prompt_continue("等 G1 完全静止 + 你准备好再进行下一个测试")

    # ── 汇总 ──
    sub.Close()
    report_summary(results)


if __name__ == "__main__":
    main()
