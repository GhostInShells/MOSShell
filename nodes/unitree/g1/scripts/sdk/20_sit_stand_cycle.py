#!/usr/bin/env python3
"""
20_sit_stand_cycle — Sit ↔ Stand 状态切换的 SDK 可达性 + 时长测量

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本(给实机执行人/未来实例)
═══════════════════════════════════════════════════════════════════════════════

用户故事的关键节点: 模型授权后"自己站起来". 这要求 SDK 能从 Sit 模式驱动 G1 站立,
不需要人按遥控器.

SDK 提供的接口:
  loco.Sit()           = SetFsmId(3)
  loco.Squat2StandUp() = SetFsmId(706)
  loco.StandUp2Squat() = SetFsmId(706)  ← 注意: SDK 源码两者映射到同一个 706
  loco.Start()         = SetFsmId(500)
  loco.Damp()          = SetFsmId(1)

未实测的关键问题:
  Q1. 从 Sit (3) 模式直接调 SetFsmId(706), G1 是否接受?
  Q2. 706 是"双向切换"(根据当前姿态自动判方向)还是"专向站立"?
  Q3. Sit → Stand 物理时长 ≈ ?
  Q4. Stand → Sit 物理时长 ≈ ?
  Q5. 站起来后 fsm_mode 自动到 Sport(6) 还是停在某个中间态? 需要额外 Start() 吗?

这些答案直接决定用户故事幕三"自己站起来"是否可行 + 模型说完话等多久切下一步.

═══════════════════════════════════════════════════════════════════════════════
执行人指引 — 你不需要动脑, 按步骤做即可
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机
  2. **G1 处于 Sit (3) 模式** — 通过遥控器 L2+某键先切到 Sit. 终端会确认
  3. 前后至少 2m 缓冲(站起来时无前后位移, 但稳妥起见)
  4. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate
  5. 遥控器在手, L2+B 兜底

测试矩阵:
  阶段 1: Sit → Squat2StandUp(706) → 观察 → 记录 mode 变化 + 时长
  阶段 2: 站立后 → StandUp2Squat(706) → 观察(看是否真能反向)
  阶段 3: 再试 Sit → Stand → Start(500) 看 mode 是否进 Sport(6)
  阶段 4: Sport → Sit 是否被允许?(直接降模式, 可能被拒)

每阶段让你打分:
  - RPC 是否成功(code == 0)?
  - 物理动作是否完成?
  - fsm_mode 变化是否符合预期?
  - 时长?

风险:
  G1 站立 / 坐下过程是 G1 自动控制, 我们只是触发. 但有"瞬间动作"可能.
  任何异常 L2+B 急停.
"""
import sys
import time
import threading
from typing import Optional


# FSM mode 解读表
FSM_NAMES = {
    0: "ZeroTorque",
    1: "Damp",
    3: "Sit",
    5: "Start/Stand(基础站立)",
    6: "Sport(运控全开)",
    # 实测中可能见到 7+, 后续补充
}


class FsmMonitor:
    """订阅 LowState 跟踪 fsm_mode 变化."""

    def __init__(self, subscriber):
        self.sub = subscriber
        self.running = False
        self.current_mode = -1
        self.mode_history: list[tuple[float, int]] = []  # (time.monotonic, mode)
        self._thread: Optional[threading.Thread] = None
        self._print = False

    def name(self, mode: int) -> str:
        return FSM_NAMES.get(mode, f"未知({mode})")

    def start(self):
        self.running = True

        def _poll():
            while self.running:
                msg = self.sub.Read(timeout=500)
                if msg is None:
                    continue
                mode = msg.mode_machine
                if mode != self.current_mode:
                    t = time.monotonic()
                    self.mode_history.append((t, mode))
                    self.current_mode = mode
                    if self._print:
                        print(f"    [{t:.2f}] fsm_mode: {self.current_mode} ({self.name(mode)})")

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def start_print(self):
        self._print = True

    def stop_print(self):
        self._print = False

    def wait_for_mode_change(self, from_mode: int, timeout: float = 15.0) -> Optional[tuple[int, float]]:
        """等待 fsm_mode 离开 from_mode. 返回 (new_mode, elapsed_seconds). 超时返回 None."""
        t_start = time.monotonic()
        while time.monotonic() - t_start < timeout:
            if self.current_mode != from_mode:
                return (self.current_mode, time.monotonic() - t_start)
            time.sleep(0.05)
        return None


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def grade_transition() -> tuple[int, str]:
    print()
    print("  对刚才的状态切换打分:")
    print("    1 = 平稳过渡, 完成度高")
    print("    2 = 完成但有小不稳/异常")
    print("    3 = 完成但很明显的抖动/警告")
    print("    4 = 失败 / 危险 / 无法完成")
    while True:
        ans = prompt("输入 1-4")
        if ans in {'1', '2', '3', '4'}:
            note = prompt("简短补充(直接回车跳过)")
            return int(ans), note
        print("  请输入 1-4")


def main():
    if len(sys.argv) < 2:
        print("用法: python 20_sit_stand_cycle.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print("=" * 70)
    print("20_sit_stand_cycle — Sit ↔ Stand SDK 可达性 + 时长")
    print("=" * 70)
    print()
    print("命题: SDK 能否驱动 G1 从坐姿站起, 再坐下?")
    print()
    print("流程: Sit → Stand → Start → Sit (中间各阶段打分)")
    print()
    print("安全:")
    print("  - 前后 2m 缓冲")
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
    print("OK")

    monitor = FsmMonitor(sub)
    monitor.start()
    monitor.start_print()
    time.sleep(1)

    print(f"\n当前 fsm_mode = {monitor.current_mode} ({monitor.name(monitor.current_mode)})")

    if monitor.current_mode != 3:
        print(f"!! 当前不在 Sit (3) 模式.")
        print("   请用遥控器先切到 Sit (通常: L2+? 组合键, 看 G1 LED 颜色).")
        prompt_continue("切到 Sit 模式后回车")
        time.sleep(1)
        if monitor.current_mode != 3:
            print(f"   仍不是 Sit (当前={monitor.current_mode}). 退出.")
            monitor.stop()
            sys.exit(1)

    results = []

    # ── 阶段 1: Sit → Stand ──
    print("\n" + "=" * 70)
    print("阶段 1: Sit (3) → Squat2StandUp(706)")
    print("=" * 70)
    print("将调用 loco.Squat2StandUp() 让 G1 站起来.")
    print("观察: G1 是否真的站起来? RPC 是否成功? fsm_mode 终态是什么?")
    prompt_continue("准备好了回车")

    t_before = time.monotonic()
    code = loco.Squat2StandUp()
    print(f"  -> Squat2StandUp() RPC code = {code}")

    if code == 0:
        change = monitor.wait_for_mode_change(from_mode=3, timeout=15.0)
        if change is not None:
            new_mode, elapsed = change
            print(f"  fsm 变化: 3 → {new_mode} ({monitor.name(new_mode)})  用时 {elapsed:.2f}s")
        else:
            print(f"  !! 15s 内未观察到 fsm_mode 变化. 当前仍是 {monitor.current_mode}")
            elapsed = -1

        # 等动作物理完成 (G1 站立通常 3-5s)
        time.sleep(3)
    else:
        elapsed = -1
        print(f"  !! RPC 被拒绝 code={code}. 可能 Sit 不能直接 706, 需要中间态.")

    grade1, note1 = grade_transition()
    results.append({
        'phase': '1. Sit → Stand (706)',
        'rpc_code': code,
        'mode_after': monitor.current_mode,
        'elapsed': elapsed,
        'grade': grade1,
        'note': note1,
    })

    if grade1 == 4:
        print("\n!!! 阶段 1 失败. 后续阶段无意义, 进入汇总.")
        monitor.stop(); sub.Close()
        print_summary(results); return

    # ── 阶段 2: Stand → Sit (反向 706) ──
    print("\n" + "=" * 70)
    print(f"阶段 2: 当前 mode = {monitor.current_mode}, 试 StandUp2Squat(706)")
    print("=" * 70)
    print("SDK 源码: StandUp2Squat 和 Squat2StandUp 都映射到 706.")
    print("如果 706 是双向的, G1 应当根据当前姿态自动坐下.")
    print("如果 706 是单向"专向站立", 这次调用应该没行为变化.")
    prompt_continue("准备好了回车")

    mode_before = monitor.current_mode
    code = loco.StandUp2Squat()
    print(f"  -> StandUp2Squat() RPC code = {code}")

    if code == 0:
        change = monitor.wait_for_mode_change(from_mode=mode_before, timeout=15.0)
        if change is not None:
            new_mode, elapsed = change
            print(f"  fsm 变化: {mode_before} → {new_mode}  用时 {elapsed:.2f}s")
        else:
            print(f"  fsm 未变化 (仍 {mode_before}) — 706 可能是单向的")
            elapsed = -1
        time.sleep(3)
    else:
        elapsed = -1
        print(f"  !! RPC 被拒绝 code={code}")

    grade2, note2 = grade_transition()
    results.append({
        'phase': '2. Stand → Sit (706 反向)',
        'rpc_code': code,
        'mode_after': monitor.current_mode,
        'elapsed': elapsed,
        'grade': grade2,
        'note': note2,
    })

    # ── 阶段 3: Sit → Stand → Start (500) ──
    print("\n" + "=" * 70)
    print(f"阶段 3: 试 Start() = SetFsmId(500), 验证能否进 Sport 模式")
    print("=" * 70)
    print("如果阶段 2 把 G1 坐下了, 这一阶段先重新站起来再 Start.")
    print("如果阶段 2 没坐下(706 单向), 直接 Start.")
    prompt_continue("准备好了回车")

    # 如果坐下了, 先站起来
    if monitor.current_mode == 3:
        print("  当前 Sit, 先 Squat2StandUp()...")
        loco.Squat2StandUp()
        time.sleep(5)

    mode_before = monitor.current_mode
    code = loco.Start()
    print(f"  -> Start() = SetFsmId(500), RPC code = {code}")

    if code == 0:
        change = monitor.wait_for_mode_change(from_mode=mode_before, timeout=15.0)
        if change is not None:
            new_mode, elapsed = change
            print(f"  fsm 变化: {mode_before} → {new_mode}  用时 {elapsed:.2f}s")
        else:
            print(f"  fsm 未变化(仍 {mode_before})")
            elapsed = -1
        time.sleep(3)
    else:
        elapsed = -1

    grade3, note3 = grade_transition()
    results.append({
        'phase': '3. Stand → Start (500)',
        'rpc_code': code,
        'mode_after': monitor.current_mode,
        'elapsed': elapsed,
        'grade': grade3,
        'note': note3,
    })

    # ── 阶段 4: Sport → Sit (降级是否允许) ──
    print("\n" + "=" * 70)
    print(f"阶段 4: 当前 mode = {monitor.current_mode}, 试 Sit() = SetFsmId(3)")
    print("=" * 70)
    print("从 Sport 直接 Sit 是否被允许? 还是要先经过中间态?")
    prompt_continue("准备好了回车")

    mode_before = monitor.current_mode
    code = loco.Sit()
    print(f"  -> Sit() = SetFsmId(3), RPC code = {code}")

    if code == 0:
        change = monitor.wait_for_mode_change(from_mode=mode_before, timeout=15.0)
        if change is not None:
            new_mode, elapsed = change
            print(f"  fsm 变化: {mode_before} → {new_mode}  用时 {elapsed:.2f}s")
        else:
            elapsed = -1
        time.sleep(3)
    else:
        elapsed = -1

    grade4, note4 = grade_transition()
    results.append({
        'phase': '4. Sport → Sit (降级)',
        'rpc_code': code,
        'mode_after': monitor.current_mode,
        'elapsed': elapsed,
        'grade': grade4,
        'note': note4,
    })

    # ── 收尾 ──
    monitor.stop()
    sub.Close()
    print_summary(results)
    print_mode_history(monitor)


def print_summary(results):
    print("\n" + "=" * 70)
    print("Sit ↔ Stand 切换汇总")
    print("=" * 70)
    print(f"\n{'阶段':<28} {'code':<5} {'终态':<5} {'用时':<7} {'分':<4} 备注")
    print("-" * 80)
    for r in results:
        elapsed_str = f"{r['elapsed']:.2f}s" if r['elapsed'] >= 0 else "—"
        print(f"{r['phase']:<28} {r['rpc_code']:<5} {r['mode_after']:<5} {elapsed_str:<7} {r['grade']:<4} {r['note']}")
    print()
    print("关键回答:")
    print("  - Q1: Sit→706 是否可达?  看阶段 1 是否 code=0 + fsm 变化")
    print("  - Q2: 706 是双向还是单向? 看阶段 2 是否能反向坐下")
    print("  - Q3: Sit→Stand 时长?     阶段 1 'elapsed'")
    print("  - Q4: Stand→Sit 时长?     阶段 2 或 4 'elapsed'")
    print("  - Q5: Start 进 Sport?     阶段 3 终态是否 6")
    print()
    print("把汇总反馈给模型实例, 模型据此更新 state DAG 设计.")


def print_mode_history(monitor):
    print("\n--- 全程 fsm_mode 变化历史 ---")
    if not monitor.mode_history:
        print("(空)")
        return
    t0 = monitor.mode_history[0][0]
    for t, mode in monitor.mode_history:
        print(f"  [{t-t0:6.2f}s] mode = {mode} ({monitor.name(mode)})")


if __name__ == "__main__":
    main()
