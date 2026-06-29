#!/usr/bin/env python3
"""
19_loco_stopmove_under_motion — 移动中 SetVelocity(0,0,0) 是否站定不仆

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本(给实机执行人/未来实例)
═══════════════════════════════════════════════════════════════════════════════

move 类命令进 warrant 的核心假设: 中断时 fallback = SetVelocity(0,0,0) 让 G1 站定.
绝对不能仆街(脱力倒地).

SDK 源码读出来 StopMove = SetVelocity(0,0,0) 不切 FSM, Sport 模式下 G1 自带平衡控制器
应当接管. 但这是源码推论, 没有实测.

本脚本要验证三件事:
  1. 移动中触发 SetVelocity(0,0,0), 真的能站定(不向前/向后冲)
  2. 站定后保持稳定(不缓慢倾倒)
  3. 站定过程中关节力矩在合理范围(不是脱力)

如果任何一条不满足:
  - 站定但有冲程 → move 的 warrant fallback 需要加"减速曲线", 不能突然归零
  - 站定后倾倒 → Sport 模式自平衡不够稳, channel 设计要重新评估
  - 力矩异常 → 关节可能瞬间脱力, 绝对不能用作 fallback

═══════════════════════════════════════════════════════════════════════════════
执行人指引 — 你不需要动脑, 按步骤做即可
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + Sport 模式
  2. 空旷场地 — G1 前后至少 2m 缓冲(本脚本最大速度 0.15 m/s, 但要预留 fallback 缓冲)
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate
  4. 遥控器在手, L2+B 始终是兜底

测试矩阵:
  3 组测试, 速度递增, 每组都是 "Move N 秒 → 立刻 SetVelocity(0,0,0) → 看站定行为"

  测试 1: vx=0.05 m/s 移动 3s → stop  — 最慢, 安全基线
  测试 2: vx=0.10 m/s 移动 3s → stop  — 中速
  测试 3: vx=0.15 m/s 移动 3s → stop  — 最快(本脚本上限)

  每次 stop 后等 5s 观察"站定后是否稳定", 然后让你打分.

观察重点(每次 stop 后):
  Q1. stop 后冲程多远(脚下到完全静止的位移)?
  Q2. 是否立刻进入平衡, 还是有摇晃?
  Q3. 站定后 5s 内是否倾倒 / 加速回退 / 异常?
  Q4. 看终端实时的 IMU rpy 是否有显著变化?

记录方法:
  脚本会同步打印 IMU rpy 实时值 + 让你对每次 stop 打分.

风险:
  最大测试速度 0.15 m/s — 慢速行走. 但仍要预留前后 2m. 任何异常 L2+B 急停.
"""
import sys
import time
import threading
import math
from typing import Optional


TEST_VELOCITIES = [
    # (vx, run_seconds, label)
    (0.05, 3.0, "慢速 0.05 m/s"),
    (0.10, 3.0, "中速 0.10 m/s"),
    (0.15, 3.0, "上限 0.15 m/s"),
]

OBSERVATION_AFTER_STOP = 5.0  # 秒, 站定后观察期


class IMUMonitor:
    """订阅 LowState, 维护最新的 IMU + 力矩快照, 后台线程跑."""

    def __init__(self, subscriber):
        self.sub = subscriber
        self.running = False
        self.rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.max_motor_tau: float = 0.0  # 全身最大力矩绝对值
        self._thread: Optional[threading.Thread] = None
        self._print = False
        self._print_counter = 0

    def start_print(self):
        self._print = True
        self._print_counter = 0

    def stop_print(self):
        self._print = False

    def start(self):
        self.running = True

        def _poll():
            while self.running:
                msg = self.sub.Read(timeout=500)
                if msg is None:
                    continue
                imu = msg.imu_state
                self.rpy = (imu.rpy[0], imu.rpy[1], imu.rpy[2])
                # 全身最大力矩
                taus = []
                for ms in msg.motor_state[:29]:  # G1 23-DoF 占 0-28
                    tau = abs(getattr(ms, 'tau_est', 0.0))
                    taus.append(tau)
                if taus:
                    self.max_motor_tau = max(taus)
                if self._print:
                    self._print_counter += 1
                    if self._print_counter % 25 == 0:  # 50ms 一次大致
                        print(f"      IMU rpy=({self.rpy[0]:+.3f}, {self.rpy[1]:+.3f}, {self.rpy[2]:+.3f})  "
                              f"max_tau={self.max_motor_tau:5.2f}")

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def grade_stop_behavior() -> tuple[int, str]:
    """提示执行人对刚才观察到的 stop 打分."""
    print()
    print("  对刚才的 SetVelocity(0,0,0) 站定行为打分:")
    print("    1 = 立刻站定, 无冲程, 稳定(理想 fallback)")
    print("    2 = 有小冲程(< 半步), 站定后稳定")
    print("    3 = 有明显冲程或站定后小幅摇晃, 但最终稳定")
    print("    4 = 站定不稳/倾倒/异常(不能用)")
    while True:
        ans = prompt("输入 1-4")
        if ans in {'1', '2', '3', '4'}:
            note = prompt("简短补充(冲程多远/有无摇晃, 没补充直接回车)")
            return int(ans), note
        print("  请输入 1-4")


def report_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("SetVelocity(0,0,0) 站定行为汇总")
    print("=" * 70)
    print()
    print(f"{'测试':<22} {'打分':<6} {'rpy 峰值':<22} {'max_tau':<10} {'备注'}")
    print("-" * 90)
    for r in results:
        rpy_str = f"({r['rpy_peak'][0]:+.3f},{r['rpy_peak'][1]:+.3f},{r['rpy_peak'][2]:+.3f})"
        print(f"{r['label']:<22} {r['grade']:<6} {rpy_str:<22} {r['max_tau']:<10.2f} {r['note']}")
    print()

    grades = [r['grade'] for r in results]
    print("结论判定:")
    if all(g == 1 for g in grades):
        verdict = "✓ 一致立刻站定 — SetVelocity(0,0,0) 可信, 可作为 move warrant 的 fallback"
    elif max(grades) <= 2:
        verdict = "~ 站定但有小冲程 — 可用, 但 channel 文档要标注'fallback 不是瞬时'"
    elif max(grades) <= 3:
        verdict = "! 部分速度下有摇晃 — move warrant 需要在高速时加减速曲线, 不能直接归零"
    else:
        verdict = "✗ 有不稳/倾倒 — move warrant 方案废弃, 需重新评估安全策略"

    print(f"  {verdict}")
    print()
    print("把这份汇总反馈给模型实例, 更新 design/2026-06-28_channel_architecture.md 'move fallback' 部分.")


def main():
    if len(sys.argv) < 2:
        print("用法: python 19_loco_stopmove_under_motion.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print("=" * 70)
    print("19_loco_stopmove_under_motion — SetVelocity(0,0,0) 站定行为验证")
    print("=" * 70)
    print()
    print("命题: move 中触发 SetVelocity(0,0,0), G1 是否站定不仆.")
    print()
    print("3 组测试, 速度 0.05 / 0.10 / 0.15 m/s 递增. 每组 Move 3s → 立刻 Stop.")
    print()
    print("安全:")
    print("  - 必须 Sport 模式")
    print("  - 前后至少 2m 缓冲, 左右 1m")
    print("  - 任何异常 L2+B")
    print("=" * 70)
    input("\n准备好了按 Enter 开始 >>> ")

    # ── 初始化 ──
    print(f"\n初始化 DDS (domain=0, interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("OK: LocoClient 就绪")

    # ── 确认 Sport ──
    print("\n确认 G1 处于 Sport 模式...")
    msg = sub.Read(timeout=2000)
    if msg is None:
        print("FAIL: LowState 收不到. 检查 DDS.")
        sys.exit(1)
    if msg.mode_machine != 6:
        print(f"!! 当前 fsm_mode = {msg.mode_machine}, 不是 Sport(6).")
        prompt_continue("用遥控器切到 Sport 后回车")
        msg = sub.Read(timeout=2000)
        if msg is None or msg.mode_machine != 6:
            print("仍不是 Sport. 退出.")
            sys.exit(1)
    print("OK: G1 处于 Sport 模式")

    monitor = IMUMonitor(sub)
    monitor.start()
    time.sleep(0.5)

    prompt_continue("最后确认: G1 前后至少 2m 缓冲, 左右 1m. 你站在 G1 后方 1m 持遥控器")

    # ── 逐组测试 ──
    results = []

    for vx, run_sec, label in TEST_VELOCITIES:
        print("\n" + "=" * 70)
        print(f"测试: {label}")
        print("=" * 70)
        print(f"  流程: Move({vx}, 0, 0) 连续模式 → 跑 {run_sec}s → SetVelocity(0,0,0)")
        print(f"        → 观察 {OBSERVATION_AFTER_STOP}s")

        prompt_continue("准备好了回车 — 然后我会启动移动")

        # 启动移动
        rpy_max = (0.0, 0.0, 0.0)
        tau_max = 0.0

        print(f"\n  -> Move({vx}, 0, 0, continuous=True)")
        monitor.start_print()
        loco.Move(vx, 0, 0, True)
        t_start = time.time()

        # 移动期记录峰值
        while time.time() - t_start < run_sec:
            r = monitor.rpy
            rpy_max = (
                max(abs(rpy_max[0]), abs(r[0])),
                max(abs(rpy_max[1]), abs(r[1])),
                max(abs(rpy_max[2]), abs(r[2])),
            )
            tau_max = max(tau_max, monitor.max_motor_tau)
            time.sleep(0.05)

        # 立刻 stop
        print(f"\n  -> SetVelocity(0, 0, 0)  [t={time.time()-t_start:.2f}s]")
        loco.SetVelocity(0.0, 0.0, 0.0)

        # 观察期记录峰值
        t_stop = time.time()
        while time.time() - t_stop < OBSERVATION_AFTER_STOP:
            r = monitor.rpy
            rpy_max = (
                max(abs(rpy_max[0]), abs(r[0])),
                max(abs(rpy_max[1]), abs(r[1])),
                max(abs(rpy_max[2]), abs(r[2])),
            )
            tau_max = max(tau_max, monitor.max_motor_tau)
            time.sleep(0.05)

        monitor.stop_print()

        print(f"\n  本次峰值: rpy_max=({rpy_max[0]:+.3f},{rpy_max[1]:+.3f},{rpy_max[2]:+.3f})  max_tau={tau_max:.2f}")

        grade, note = grade_stop_behavior()

        results.append({
            'label': label,
            'vx': vx,
            'rpy_peak': rpy_max,
            'max_tau': tau_max,
            'grade': grade,
            'note': note,
        })

        if grade == 4:
            print("\n  !!! 打分 4 = 不稳/倾倒. 立刻终止后续测试.")
            break

        prompt_continue("等 G1 完全静止 + 你准备好再进行下一组")

    monitor.stop()
    sub.Close()

    # ── 汇总 ──
    report_summary(results)


if __name__ == "__main__":
    main()
