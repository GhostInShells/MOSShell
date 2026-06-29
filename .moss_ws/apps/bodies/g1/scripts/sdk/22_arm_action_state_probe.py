#!/usr/bin/env python3
"""
22_arm_action_state_probe — 探测 rt/arm/action/state topic 的内容

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

2026-06-16 session 实测确认 G1 发布 rt/arm/action/state topic, 但**未实测其内容**.

如果这个 topic 包含 "in_progress / done" 标志, channel 实现可以:
  arm.execute_action() → 等 state topic 报 done → 命令的 await 返回
这解决了 "ExecuteAction 是非阻塞 RPC, channel await 时长 ≈ ? " 的核心问题.

如果 state topic 内容不直接报完成状态, 备选方案:
  - 看 joints 速度趋零(慢, 但通用)
  - 预估时长 sleep(粗, 但简单)

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + Sport 模式
  2. 手臂 1m 半径无人无物
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate

测试流程:
  阶段 1: 静默订阅 5s — 看 state topic 在没动作时发什么(可能是周期性心跳)
  阶段 2: 触发 face wave(25) — 同步记录所有 state 变化, 看从触发到物理完成的整段
  阶段 3: 触发 hands up(15) + 中途 release — 看中断时 state 报什么

每条 state 消息内容都会原样打印, 你不用动脑, 看完终端复制内容反馈即可.

注意 unitree_sdk2_python 中 rt/arm/action/state 的消息类型未知, 我们先尝试
几种已知类型(String_/JointState_/Empty_)做反射, 如果都不匹配会打印原始字节供分析.

风险:
  和 18 一样, arm 动作有破坏性, 周围 1m 无物.
"""
import sys
import time
import threading
from typing import Optional

# 候选 message type 列表 — 按可能性高低排
# rt/arm/action/state 的真实类型未知, 我们尝试这些
CANDIDATE_TYPES = [
    # (import_path, type_name, description)
    ("unitree_sdk2py.idl.std_msgs.msg.dds_", "String_", "JSON string (常见)"),
    ("unitree_sdk2py.idl.unitree_hg.msg.dds_", "MotorState_", "单关节状态"),
    ("unitree_sdk2py.idl.unitree_go.msg.dds_", "Go2FrontVideoData_", "兜底尝试"),
]


def try_import(import_path: str, type_name: str):
    """尝试 import 一个类型, 失败返回 None."""
    try:
        module = __import__(import_path, fromlist=[type_name])
        return getattr(module, type_name)
    except (ImportError, AttributeError):
        return None


def msg_summary(msg) -> str:
    """把消息对象打印成可读字符串."""
    if msg is None:
        return "None"

    parts = []
    # 试图打印 dataclass 字段
    if hasattr(msg, '__dict__'):
        for k, v in msg.__dict__.items():
            if k.startswith('_'):
                continue
            val_str = str(v)[:80]
            parts.append(f"{k}={val_str}")
        return "  ".join(parts) if parts else repr(msg)
    return repr(msg)


class StateMonitor:
    def __init__(self, subscriber):
        self.sub = subscriber
        self.running = False
        self.messages: list[tuple[float, str]] = []  # (t, summary)
        self._thread: Optional[threading.Thread] = None
        self._print = False
        self._start_t = 0.0

    def start(self):
        self.running = True
        self._start_t = time.monotonic()

        def _poll():
            while self.running:
                msg = self.sub.Read(timeout=200)
                if msg is None:
                    continue
                t = time.monotonic() - self._start_t
                summary = msg_summary(msg)
                self.messages.append((t, summary))
                if self._print:
                    print(f"    [{t:6.2f}s] {summary}")

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def reset(self):
        self.messages.clear()
        self._start_t = time.monotonic()

    def enable_print(self):
        self._print = True


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def main():
    if len(sys.argv) < 2:
        print("用法: python 22_arm_action_state_probe.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    print("=" * 70)
    print("22_arm_action_state_probe — rt/arm/action/state 内容探测")
    print("=" * 70)
    print()
    print("命题: rt/arm/action/state 是否含 'in_progress / done' 状态?")
    print()
    input("准备好了按 Enter 开始 >>> ")

    # ── 找出 topic 的消息类型 ──
    print(f"\n初始化 DDS (interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    print("\n尝试候选消息类型订阅 rt/arm/action/state...")
    state_sub = None
    state_type_name = None
    for import_path, type_name, desc in CANDIDATE_TYPES:
        cls = try_import(import_path, type_name)
        if cls is None:
            print(f"  {type_name:<30} import 失败, 跳过")
            continue
        try:
            sub = ChannelSubscriber("rt/arm/action/state", cls)
            sub.Init()
            # 试读 1s, 看能否拿到数据 / 是否抛错
            msg = sub.Read(timeout=1000)
            if msg is not None:
                print(f"  {type_name:<30} ✓ 收到数据! 用此类型")
                print(f"      首帧 sample: {msg_summary(msg)[:200]}")
                state_sub = sub
                state_type_name = type_name
                break
            else:
                print(f"  {type_name:<30} 订阅成功但 1s 内无数据, 关闭")
                sub.Close()
        except Exception as e:
            print(f"  {type_name:<30} 异常: {e}")

    if state_sub is None:
        print("\n!! 所有候选类型都未成功 — 需要进一步分析.")
        print("   建议: 用 cyclonedds CLI 检查 rt/arm/action/state 的 IDL:")
        print("     source /etc/profile.d/cyclonedds.sh")
        print("     cyclonedds ts rt/arm/action/state")
        print("   把输出反馈给模型, 模型会找到正确的 type import path.")
        sys.exit(1)

    print(f"\n✓ 使用消息类型 {state_type_name}")

    # ── LowState 订阅(用来确认 Sport) ──
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init()

    msg = lowstate_sub.Read(timeout=2000)
    if msg is None or msg.mode_machine != 6:
        fsm = msg.mode_machine if msg else 'None'
        print(f"\n!! 当前 fsm = {fsm}, 不是 Sport(6)")
        prompt_continue("切到 Sport 后回车")
        msg = lowstate_sub.Read(timeout=2000)
        if msg is None or msg.mode_machine != 6:
            print("仍不是 Sport. 退出.")
            sys.exit(1)
    print(f"OK: Sport 模式")

    # ── ArmClient ──
    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()

    monitor = StateMonitor(state_sub)
    monitor.start()
    monitor.enable_print()

    # ── 阶段 1: 静默订阅 5s ──
    print("\n" + "=" * 70)
    print("阶段 1: 静默订阅 5s — 看心跳/空闲时的消息")
    print("=" * 70)
    monitor.reset()
    time.sleep(5)
    idle_count = len(monitor.messages)
    print(f"\n  空闲期收到 {idle_count} 条消息")

    # ── 阶段 2: 触发 face wave, 完整观察 ──
    print("\n" + "=" * 70)
    print("阶段 2: ExecuteAction(25) face wave — 完整记录")
    print("=" * 70)
    prompt_continue("准备好了回车触发 face wave")

    monitor.reset()
    print(f"  [0.00s] 调用 ExecuteAction(25)")
    code = arm.ExecuteAction(25)
    print(f"  RPC code = {code}")

    print(f"  观察 6s 让动作完成...")
    time.sleep(6)
    wave_count = len(monitor.messages)
    print(f"\n  face wave 期间收到 {wave_count} 条消息")

    # release 收尾
    print(f"  发 release 收尾...")
    monitor.reset()
    arm.ExecuteAction(99)
    time.sleep(3)
    release_count = len(monitor.messages)
    print(f"\n  release 期间收到 {release_count} 条消息")

    # ── 阶段 3: hands up + 中途 release ──
    print("\n" + "=" * 70)
    print("阶段 3: hands up 中途打断 — 看中断时 state 报什么")
    print("=" * 70)
    prompt_continue("准备好了回车")

    monitor.reset()
    print(f"  [0.00s] ExecuteAction(15) hands up")
    arm.ExecuteAction(15)
    time.sleep(1.5)
    print(f"  [{time.monotonic() - monitor._start_t:.2f}s] ExecuteAction(99) release 打断")
    arm.ExecuteAction(99)
    time.sleep(4)
    interrupt_count = len(monitor.messages)
    print(f"\n  hands up + release 期间收到 {interrupt_count} 条消息")

    # ── 收尾 ──
    monitor.stop()
    state_sub.Close()
    lowstate_sub.Close()

    print("\n" + "=" * 70)
    print("观测汇总")
    print("=" * 70)
    print(f"  消息类型:     {state_type_name}")
    print(f"  空闲期消息数: {idle_count} (5s)")
    print(f"  face wave:    {wave_count} (6s)")
    print(f"  release:      {release_count} (3s)")
    print(f"  hands+inter:  {interrupt_count} (5.5s)")
    print()
    print("解读指南(反馈给模型):")
    print("  - 空闲期数 ≈ 0  → state 是事件型(动作时才发)")
    print("  - 空闲期数 > 0  → state 是周期型(总在发, 字段表状态)")
    print("  - 不同阶段的消息字段是否能识别出 'in_progress / done'?")
    print("  - 如果是字符串型, 内容是否是 JSON, 字段名是什么?")
    print()
    print("把上面每阶段打印的消息原文反馈给模型, 模型据此设计 channel 等待逻辑.")


if __name__ == "__main__":
    main()
