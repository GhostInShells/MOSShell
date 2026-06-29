#!/usr/bin/env python3
"""
22_arm_action_state_probe — 探测 rt/arm/action/state topic 的内容

前次 06-16 确认该 topic 存在, 类型为 std_msgs/msg/String_ (JSON).
本脚本看 JSON 里是否有 action 完成状态 (in_progress / done).

用法:
  python scripts/sdk/22_arm_action_state_probe.py <networkInterface>
  python scripts/sdk/22_arm_action_state_probe.py eth0

测试:
  1. 订阅 topic, 空闲 3s — 看有没有心跳
  2. 发 face wave (25), 记录所有消息 — 看是否有 state 变化
  3. 发 clap (17) + release (99) — 看 state 过渡

实测记录:
  2026-06-29: 首次实机运行.
"""
import sys
import time
import threading
import json
from typing import Optional


def main():
    if len(sys.argv) < 2:
        print("用法: python 22_arm_action_state_probe.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    print("=" * 70)
    print("22_arm_action_state_probe — rt/arm/action/state 内容探测")
    print("=" * 70)
    print(f"\n初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/arm/action/state", String_)
    sub.Init()

    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()

    print("就绪")

    # ── 收集线程 ──
    _messages: list[tuple[float, dict]] = []
    _lock = threading.Lock()
    running = True

    def _poll():
        while running:
            msg = sub.Read(timeout=500)
            if msg is None:
                continue
            try:
                data = json.loads(msg.data) if msg.data else {}
            except json.JSONDecodeError:
                data = {"_raw": str(msg.data)[:120]}
            with _lock:
                _messages.append((time.monotonic(), data))
                # 只打印摘要, 不刷屏
                keys = [k for k in data if not k.startswith('_')]
                print(f"  [{_messages[-1][0]:.1f}s] keys={keys}  sample={str(data)[:200]}")

    _thread = threading.Thread(target=_poll, daemon=True)
    _thread.start()
    time.sleep(0.5)

    _t0 = time.monotonic()

    def collect(label: str, wait: float) -> list[dict]:
        """等 wait 秒, 返回这段时间内收集的消息."""
        time.sleep(wait)
        with _lock:
            msgs = [m for m in _messages if m[0] >= _t0]
        print(f"\n  [{label}] {len(msgs)} 条消息 ({wait}s 内)")
        return [m[1] for m in msgs]

    def trigger(action_id: int, name: str):
        print(f"\n--- ExecuteAction({action_id}) [{name}] ---")
        code = arm.ExecuteAction(action_id)
        print(f"  RPC code = {code}")

    # ── 阶段 1: 空闲 3s ──
    print("\n" + "=" * 70)
    print("阶段 1: 空闲 3s — 看心跳")
    print("=" * 70)
    msgs = collect("idle", 3.0)

    # ── 阶段 2: face wave 完整周期 ──
    print("\n" + "=" * 70)
    print("阶段 2: face wave 完整录制")
    print("=" * 70)
    input("准备好了按 Enter 触发 face wave >>> ")
    _t0 = time.monotonic()
    trigger(25, "face wave")
    msgs = collect("face wave", 5.0)

    # ── 阶段 3: clap + release ──
    print("\n" + "=" * 70)
    print("阶段 3: clap + release")
    print("=" * 70)
    input("准备好了按 Enter >>> ")
    _t0 = time.monotonic()
    trigger(17, "clap")
    time.sleep(1.0)
    trigger(99, "release")
    msgs = collect("clap+release", 5.0)

    # ── 收尾 ──
    running = False
    _thread.join(timeout=2)
    sub.Close()

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("把上面打印的消息 key+sample 反馈给模型.")
    print("模型判断: topic 是否包含 in_progress/done 状态, 用于 arm 命令 await 实现.")


if __name__ == "__main__":
    main()
