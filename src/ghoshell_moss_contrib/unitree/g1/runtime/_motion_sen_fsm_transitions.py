"""
_motion_sen_fsm_transitions — Motion FSM 转换实时打印 + Enter drain.

场景:
  操作 G1 切换运动模式 (落座 / 站立 / 运控), 看 motion runtime 是否能
  及时捕获 fsm_mode 变化为 MotionTransition 入 ring buffer.
  listener 后台实时打印每次 transition, 主线程按 Enter 触发 drain
  看 batch 形态 (current + transitions + window_seconds).

  这是 channel 真实使用 scenario 的最小模拟:
    "FSM 模式是离散事件, channel 按命令周期拿历史 transitions
     一次性 dump 给模型, 不持续 push 上下文."

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._motion_sen_fsm_transitions <nic>

  nic 示例: eth0 / enp3s0 — 见 docs/hardware.md.

前置:
  - G1 已开机, 任何模式都可以 (脚本以开机首次状态为 baseline)
  - 准备好遥控器, 计划在 30s ~ 1min 内做以下任一序列:
      a) 阻尼 (L2+B 急停) → 长按 L2+A 切换运控  → 站立 / 行走
      b) 已运控 → 长按 L2+左 落座 → 长按 L2+A 长按切阻尼
  - **不进调试模式** (story-2026-07.md §0: Sport → L2+R2 会触发 PC1 保护故障)
  - 吊架下操作, 任何站立/落座切换需空间安全

预期:
  [motion#1] Sit → Start  fsm_id 3→5  at T+12.3s
  [motion#2] Start → Sport  fsm_id 5→6  at T+13.5s
  ...
  >>> press Enter to drain >>>
  [drain] current=Sport (fsm_id=6) window=28.4s transitions=2
    T-16.1s: Sit → Start
    T-14.9s: Start → Sport
  [health] {'running': True, 'last_seen_fsm_mode': 6, ...}

  Ctrl+C 退出 → motion.stop() + unregister + 摘要.

读完 docstring 还看不懂请回去读 runtime/README.md.
"""
from __future__ import annotations

import sys
import time

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import motion
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


_count = 0
_started_at = 0.0


def _on_transition(t: motion.MotionTransition) -> None:
    """每次 fsm_mode 变化触发. 跑在 poller 线程."""
    global _count
    _count += 1
    rel = t.at - _started_at
    print(
        f"[motion#{_count}] {t.from_name} → {t.to_name}  "
        f"fsm_id {t.from_mode}→{t.to_mode}  at T+{rel:.1f}s"
    )


def _format_batch(batch: motion.MotionHistoryBatch) -> str:
    lines = [
        f"[drain] current={batch.current.mode_name} "
        f"(fsm_id={batch.current.fsm_mode}) "
        f"window={batch.window_seconds:.1f}s "
        f"transitions={len(batch.transitions)}",
    ]
    if batch.transitions:
        for t in batch.transitions:
            rel = t.at - batch.current.captured_at
            lines.append(
                f"  T{rel:+.1f}s: {t.from_name} → {t.to_name} "
                f"(fsm {t.from_mode} → {t.to_mode})"
            )
    else:
        lines.append("  (no transitions in window)")
    return "\n".join(lines)


def main(nic: str) -> int:
    global _started_at

    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)
    _started_at = time.time()

    print("[2/3] motion.start() ...")
    motion.start()
    handle = motion.register_listener(_on_transition)
    baseline = motion.read_current()
    print(f"      baseline = {baseline.mode_name} (fsm_id={baseline.fsm_mode})")
    print(f"      listener handle = {handle}")

    print()
    print("=" * 64)
    print(" 用遥控器切换 FSM 模式, 看 [motion#N] 实时打印.")
    print(" 推荐序列 (吊架下操作, 不进调试模式):")
    print("   - L2+B 急停 → 长按 L2+A 切运控 → 站立 / 行走")
    print("   - 已运控时 → 长按 L2+左 落座")
    print(" 按 Enter   → drain 当前 buffer (current + transitions + health)")
    print(" Ctrl+C    → 干净退出")
    print("=" * 64)
    print()

    session: PromptSession = PromptSession()
    drain_count = 0
    try:
        with patch_stdout.patch_stdout(raw=True):
            while True:
                try:
                    session.prompt(">>> press Enter to drain >>> ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                drain_count += 1
                batch = motion.drain()
                print(_format_batch(batch))
                print(f"[health] {motion.health()}\n")
    finally:
        print(f"\n[3/3] motion.stop() ...")
        motion.unregister_listener(handle)
        motion.stop()
        print()
        print("=" * 64)
        print(f" 摘要: transitions {_count} 次, drain {drain_count} 次.")
        print("=" * 64)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
