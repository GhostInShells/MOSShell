"""
_asr_sen_listen_and_drain — ASR 双工体验: listener 实时打印 + Enter drain.

场景:
  你启动脚本后, 对着 G1 说话. ASR listener 在后台把每条识别结果实时打印出来,
  你随时按 Enter 触发 asr.drain() —— 看本次 drain 拿到几条 + forgotten 多少 + health 状态.
  这是 channel 真实使用 senario 的最小模拟:
    "listener 维持感知, 命令时机 drain 入上下文".

  脚本同时回答一个设计问题:
    callback 拿到的 result, 是否总是 == 那一刻的 peek_latest()?
    每条 listener 行会标 `peek_matches=True/False`, 跑一会儿就知道了.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._asr_sen_listen_and_drain <nic>

  nic 示例: eth0 / enp3s0 — 见 docs/hardware.md.

前置:
  - G1 已开机 (任何模式; ASR 不依赖运动模式)
  - PC2 mic 链路 ok (G1 内置 mic, 出厂可用)
  - 你身处 G1 前方 1-2m, 准备说几句中文 + 英文

预期:
  [asr] 'xxx'  speaker=N  angle=N  final=True  peek_matches=True       ← 实时刷新
  ...
  >>> press Enter to drain >>>                                          ← 输入框稳定在底部
  [drain] items=3 forgotten=0
    [1] 'xxx'  ts=...
    ...
  [health] {'running': True, 'buffer_len': 0, ...}

  Ctrl+C 退出 → 自动 asr.stop() + unregister_listener + 摘要.

实测样本 (2026-07-01, 小声说话):
  [asr#1] '你好啊。'  speaker=0  angle=0  final=False
  [asr#2] '我现在在机器人的右侧。'  speaker=0  angle=0  final=False
  drain 正常取走 items, forgotten=0. 空 drain 返回 items=0.
  物理事实: angle 始终 0 (G1 默认不启用声源定位); is_final 始终 false (非流式);
  speaker_id 始终 0 (不启用说话人识别). VAD 判停时 LED 变色但无 DDS 信号.

读完 docstring 还看不懂请回去读 runtime/README.md.
"""
from __future__ import annotations

import sys

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import asr
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


# ── listener: 跑在 reader 线程, 不能阻塞 ──────────────────────────────────

_listener_count = 0  # 仅 listener 线程读写, 不需要锁


def _on_asr_result(result: asr.AsrResult) -> None:
    """ASR 新数据到达时触发. 立刻 print + 跟 peek_latest 对比."""
    global _listener_count
    _listener_count += 1

    latest = asr.peek_latest()
    peek_matches = latest is not None and latest.id == result.id

    # patch_stdout 包裹下 print 不会破坏底部 prompt 输入框.
    print(
        f"[asr#{_listener_count}] {result.text!r}  "
        f"speaker={result.speaker_id}  angle={result.angle}  "
        f"final={result.is_final}  peek_matches={peek_matches}"
    )


def _format_batch(batch: asr.AsrBatch) -> str:
    lines = [f"[drain] items={len(batch.items)} forgotten={batch.forgotten}"]
    for i, r in enumerate(batch.items, 1):
        lines.append(f"  [{i}] {r.text!r}  ts={r.received_at:.2f}  speaker={r.speaker_id}")
    return "\n".join(lines)


def main(nic: str) -> int:
    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)

    print("[2/3] asr.start() ...")
    asr.start(buffer_size=32)
    handle = asr.register_listener(_on_asr_result)
    print(f"      listener handle = {handle}")

    print()
    print("=" * 64)
    print(" 对着 G1 说话, 看 [asr#N] 行实时刷新.")
    print(" 按 Enter   → drain 当前 buffer (含 forgotten 计数 + health)")
    print(" Ctrl+C    → 干净退出 (stop + unregister + 摘要)")
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
                batch = asr.drain()
                print(_format_batch(batch))
                print(f"[health] {asr.health()}\n")
    finally:
        print(f"\n[3/3] asr.stop() ...")
        asr.unregister_listener(handle)
        asr.stop()
        print()
        print("=" * 64)
        print(f" 摘要: listener 触发 {_listener_count} 次, drain {drain_count} 次.")
        print("=" * 64)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
