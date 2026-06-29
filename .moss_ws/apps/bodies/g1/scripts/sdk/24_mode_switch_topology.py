#!/usr/bin/env python3
"""
24_mode_switch_topology — FSM 模式完整可达图实测

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

20 脚本测了 Sit / Stand / Sport 路径中的关键边. 24 是它的补完 —
对所有已知 FSM mode 之间的 SetFsmId(X) 调用做系统性遍历,
得到完整的"哪些边可达 / 哪些被拒 / 各边耗时"的可达图.

state DAG 设计的"具体边定义"依赖这张图. 没这张图就只能保守地假设
"任何边都要先 Damp 再切", 用户体验差.

已知 mode (FSM ID, 来自 docs/index.md + SDK 源码):
  0 = ZeroTorque  (危险, 永久封禁)
  1 = Damp        (急停阻尼)
  3 = Sit         (落座)
  500 = Start     (基础站立)
  706 = Squat2StandUp / StandUp2Squat (双向?)
  Sport mode_machine=6 — 是否对应特定 FSM ID? Sport 是怎么进的?

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机, 当前 fsm_mode 自动检测(脚本会从 Damp 开始)
  2. 前后 2m 缓冲, 周围 1m 无物 (G1 可能站起/坐下/摆姿态)
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate

测试矩阵:
  按"从安全态开始, 逐步往复杂态切"的顺序遍历, 每条边都记录:
  - SetFsmId(target) 的 RPC code
  - fsm_mode 实际变化值
  - 时长
  - 物理稳定性(你打分)

测试顺序:
  Damp(1) → Sit(3) → Sit→706 → 站起后 fsm_mode → Start(500) → Sport?
  → 各种"非法"边: Sport→Sit直接 / Damp→500 直接 等
  → 收尾 Damp

风险:
  状态切换有体姿变化. 任何不稳 L2+B 急停.
"""
import sys
import time
import threading
from typing import Optional


FSM_NAMES = {
    0: "ZeroTorque",
    1: "Damp",
    3: "Sit",
    5: "Start/Stand",
    6: "Sport",
}


# 测试边: (from_label, target_fsm_id, description, expected_safe)
# 注意 — 没有"等待到达 from"的预设, 调用前要确保当前是 from
TEST_EDGES = [
    ("Damp(1) → Sit(3)",       3,   "从急停降到坐", True),
    ("Sit(3) → 706 (双向?)",    706, "Sit 模式调 706 看是否站起", True),
    ("Stand → Start(500)",     500, "站立后调 Start 是否进 Sport", True),
    ("Sport → Sit(3) 直接",     3,   "Sport 直接降 Sit (可能被拒)", False),
    ("Sit → Start(500) 跳跃",   500, "Sit 直接 Start, 不经过站立 (可能被拒)", False),
    ("当前 → 706",              706, "在最后状态调 706, 看是否反向回坐", False),
    ("收尾 → Damp(1)",          1,   "测试完毕回到 Damp", True),
]


class FsmMonitor:
    def __init__(self, subscriber):
        self.sub = subscriber
        self.running = False
        self.current_mode = -1
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        def _poll():
            while self.running:
                msg = self.sub.Read(timeout=500)
                if msg is None:
                    continue
                self.current_mode = msg.mode_machine
        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def wait_for_change(self, from_mode: int, timeout: float = 15.0) -> Optional[tuple[int, float]]:
        t_start = time.monotonic()
        while time.monotonic() - t_start < timeout:
            if self.current_mode != from_mode:
                return (self.current_mode, time.monotonic() - t_start)
            time.sleep(0.05)
        return None

    def name(self, mode: int) -> str:
        return FSM_NAMES.get(mode, f"未知({mode})")


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def grade_transition() -> tuple[int, str]:
    print()
    print("  状态切换打分:")
    print("    1 = 顺利完成, 终态稳定")
    print("    2 = 完成但有小不稳")
    print("    3 = 完成但抖动明显")
    print("    4 = 被拒 / 不可完成 / 危险")
    while True:
        ans = prompt("输入 1-4")
        if ans in {'1', '2', '3', '4'}:
            note = prompt("简短补充")
            return int(ans), note
        print("  请输入 1-4")


def main():
    if len(sys.argv) < 2:
        print("用法: python 24_mode_switch_topology.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print("=" * 70)
    print("24_mode_switch_topology — FSM 完整可达图")
    print("=" * 70)
    print()
    input("准备好了按 Enter 开始 >>> ")

    print(f"\n初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()

    monitor = FsmMonitor(sub)
    monitor.start()
    time.sleep(1)

    print(f"\n当前 fsm_mode = {monitor.current_mode} ({monitor.name(monitor.current_mode)})")

    if monitor.current_mode != 1:
        print(f"!! 当前不在 Damp(1). 请用遥控器先切到 Damp.")
        prompt_continue("切到 Damp 后回车")
        if monitor.current_mode != 1:
            print(f"   仍不是 Damp (当前={monitor.current_mode}). 退出.")
            monitor.stop()
            sys.exit(1)

    print(f"\nOK: 起点 Damp(1)")

    results = []

    for (label, target, desc, safe) in TEST_EDGES:
        print("\n" + "=" * 70)
        print(f"边: {label}")
        print(f"说明: {desc}")
        print(f"当前 fsm = {monitor.current_mode} ({monitor.name(monitor.current_mode)})")
        if not safe:
            print("⚠️  这是可能"被拒"的边 — 期望它被 G1 拒绝或行为异常")
        print("=" * 70)
        prompt_continue("准备好了回车")

        mode_before = monitor.current_mode
        t_before = time.monotonic()
        code = loco.SetFsmId(target)
        print(f"  -> SetFsmId({target}) RPC code = {code}")

        if code == 0:
            change = monitor.wait_for_change(from_mode=mode_before, timeout=15.0)
            if change is not None:
                new_mode, elapsed = change
                print(f"  fsm 变化: {mode_before} → {new_mode} ({monitor.name(new_mode)})  用时 {elapsed:.2f}s")
            else:
                new_mode = monitor.current_mode
                elapsed = -1
                print(f"  fsm 未变化(仍 {new_mode})")
            time.sleep(2)
        else:
            new_mode = monitor.current_mode
            elapsed = -1
            print(f"  !! 被拒")

        grade, note = grade_transition()

        results.append({
            'label': label,
            'target': target,
            'mode_before': mode_before,
            'mode_after': new_mode,
            'rpc_code': code,
            'elapsed': elapsed,
            'grade': grade,
            'note': note,
        })

        if grade == 4 and safe:
            print("\n!!! 预期安全的边失败 — 终止后续测试.")
            break

    monitor.stop()
    sub.Close()

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("FSM 可达图汇总")
    print("=" * 70)
    print()
    print(f"{'边':<28} {'from':<5} {'target':<7} {'code':<5} {'after':<6} {'用时':<8} {'分':<4} 备注")
    print("-" * 90)
    for r in results:
        elapsed_str = f"{r['elapsed']:.2f}s" if r['elapsed'] >= 0 else "—"
        print(f"{r['label']:<28} {r['mode_before']:<5} {r['target']:<7} {r['rpc_code']:<5} "
              f"{r['mode_after']:<6} {elapsed_str:<8} {r['grade']:<4} {r['note']}")
    print()
    print("反馈给模型: 模型据此画出完整 state DAG 边定义.")


if __name__ == "__main__":
    main()
