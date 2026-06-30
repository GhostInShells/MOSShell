"""
_control_pad_sen_listen_and_drain — control_pad 双工体验.

场景:
  启动后注册几个 G1 物理无影响的按键 binding (f1 / f3 / start + 组合).
  你按遥控器对应键, event listener 在后台实时打印 [event] / [fallthrough] 行.
  随时按 Enter 触发 control_pad.drain() — 看本次 drain 拿到几条 + forgotten + health.

  也可以快速连按同一键测试 debounce — 200ms 内连按只触发一次.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_sen_listen_and_drain <nic>

  nic 示例: eth0 / enp3s0 — 见 docs/hardware.md.

前置:
  - G1 已开机 (任何模式, 本脚本不触发任何 G1 物理动作)
  - 你身处遥控器手够得着的距离

安全:
  - 演示用键全部是 Ghost 控制键 (f1 / f3 / start), G1 主板不响应它们
  - 千万 *不要* 按 L2+B (硬件急停, G1 立刻进 Damp, 吊架上凌空蹬腿)
  - 不要按 L2+R2 (从 Sport 进调试模式会触发 PC1 保护性故障)
  - 不要推摇杆 (运动模式下 G1 会动)

演示 binding (G1 物理无影响):
  ghost_trigger      = {f1}
  ghost_interrupt    = {f3}
  channel_interrupt  = {start}
  combo_f1_f3        = {f1, f3}           # 组合键演示, 最后按下的键边沿命中

  select 故意不注册 — 按 select → fallthrough listener 触发.
  快速连按同一键 → 看 [event] 只出现一次 (debounce 命中).

预期:
  [event#1] ghost_trigger      keys=f1
  [event#2] FALLTHROUGH        keys=select
  [event#3] combo_f1_f3        keys=f1+f3
  >>> press Enter to drain >>>
  [drain] items=3 forgotten=0
    [1] ghost_trigger      keys=f1            ts=...
    [2] FALLTHROUGH        keys=select        ts=...
    [3] combo_f1_f3        keys=f1+f3         ts=...
  [health] {'running': True, 'bindings_count': 4, ...}

  Ctrl+C → 干净退出 (stop + unregister + 摘要).

读完 docstring 还看不懂请回去读 runtime/README.md 和 control_pad.py 顶部 docstring.
"""
from __future__ import annotations

import sys

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


# ── listener: 跑在 sdk reader 线程, 不能阻塞 ─────────────────────────────

_event_count = 0


def _on_event(event: control_pad.KeyEvent) -> None:
    """任何事件 (binding 命中 OR fallthrough) 都通知. 走 event listener
    比给每个 binding 配 cb 更省, 显示也统一."""
    global _event_count
    _event_count += 1
    label = "FALLTHROUGH" if event.is_fallthrough else event.binding_name
    keys_str = "+".join(event.pressed_keys)
    print(f"[event#{_event_count}] {label:<20}  keys={keys_str}")


def _noop(_event: control_pad.KeyEvent) -> None:
    """binding cb 占位 — 我们走统一的 event listener, 不在 per-binding cb 里 print."""
    pass


def _format_batch(batch: control_pad.KeyEventBatch) -> str:
    lines = [f"[drain] items={len(batch.items)} forgotten={batch.forgotten}"]
    for i, e in enumerate(batch.items, 1):
        label = "FALLTHROUGH" if e.is_fallthrough else e.binding_name
        keys_str = "+".join(e.pressed_keys)
        lines.append(
            f"  [{i}] {label:<20}  keys={keys_str:<12}  ts={e.triggered_at:.2f}"
        )
    return "\n".join(lines)


def main(nic: str) -> int:
    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)

    print("[2/3] control_pad.start() + register bindings ...")
    control_pad.start(buffer_size=32)

    handles: list[str] = []
    handles.append(control_pad.register_binding("ghost_trigger", {"f1"}, _noop))
    handles.append(control_pad.register_binding("ghost_interrupt", {"f3"}, _noop))
    handles.append(control_pad.register_binding("channel_interrupt", {"start"}, _noop))
    handles.append(control_pad.register_binding("combo_f1_f3", {"f1", "f3"}, _noop))

    event_handle = control_pad.register_event_listener(_on_event)

    print()
    print("=" * 72)
    print(" Bindings: ghost_trigger(f1), ghost_interrupt(f3),")
    print("           channel_interrupt(start), combo_f1_f3(f1+f3)")
    print()
    print(" 按 select   → fallthrough (无 binding 匹配)")
    print(" 快速连按同一键 → debounce 静默 (默认 200ms cooldown)")
    print(" 按 Enter    → drain 当前 buffer (含 forgotten + health)")
    print(" Ctrl+C     → 干净退出 (stop + unregister + 摘要)")
    print()
    print(" 安全: 不要按 L2+B / L2+R2; 不要推摇杆")
    print("=" * 72)
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
                batch = control_pad.drain()
                print(_format_batch(batch))
                print(f"[health] {control_pad.health()}\n")
    finally:
        print("\n[3/3] cleanup ...")
        for h in handles:
            control_pad.unregister_binding(h)
        control_pad.unregister_event_listener(event_handle)
        control_pad.stop()
        print()
        print("=" * 72)
        print(f" 摘要: event 触发 {_event_count} 次, drain {drain_count} 次.")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
