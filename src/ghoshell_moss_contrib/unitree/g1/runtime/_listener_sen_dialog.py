"""
_listener_sen_dialog — listener 端到端验证: 流式 partial + 按键打断 drain.

场景:
  跑起来后, listener.start() 默认从 ~/.moss_g1_listener.json 读配置, 自动起 capture
  + VolcengineASR ws. 你戴蓝牙耳机说话:

    - partial 在 ASR 服务端边识别边送来, 这里**实时刷新一行**, 看到 text 不断长大
    - 服务端 VAD 判停后 (end_window_ms 静音), 触发 sentence_listener, 这里打印一行
      [FINAL #N] ...
    - 你按 Enter 触发 drain() — 把 finalized buffer 全部拿走, 显示一批
    - 你输入 f + Enter 触发 drain(force_finalize_partial=True) — 把当前正在说的 partial
      也强制当 final 拿走. 这是真实交互里 "按遥控器键打断 ASR VAD" 的入口.
    - 你输入 h + Enter 看完整 health snapshot
    - 故意拔蓝牙 / 拔耳机 / 重连, 看 health 状态切换 + health_change_listener 行
    - Ctrl+C 干净退出: stop + 摘要

  这个脚本回答了三件事:
    1) listener 启动时蓝牙未连 / 配置不存在的 fallback 是否真的不抛
    2) 流式 partial 高频刷新 + force drain 互斥是否正确
    3) 蓝牙断连后 listener 能否自动重连, health 是否准确反映

Usage:
  # 默认走 ~/.moss_g1_listener.json. 没生成过先跑 _listener_sen_setup.
  python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_dialog

  python -m ghoshell_moss_contrib.unitree.g1.runtime._listener_sen_dialog \
      --config /custom/path.json

前置:
  - 已运行 _listener_sen_setup 生成 ~/.moss_g1_listener.json
  - 蓝牙耳机已连且能录音 (setup 阶段 voiced 占比 > 5%)
  - .moss_ws/.env 含 VOLCENGINE_BM_ASR_APPID + VOLCENGINE_BM_ASR_TOKEN
    (脚本自动从 cwd 或 .moss_ws/ 加载)

预期输出 (示意):
  [bootstrap] listener.start() ...
  [bootstrap] health status=ok  device=AirPods Pro (Hands-Free)  sr=16000

  >>> Enter=drain  /  f+Enter=force drain  /  h+Enter=health  /  Ctrl+C=stop >>>

  [partial] 你好我是              ← 同一行覆盖刷新
  [partial] 你好我是 MOSS         ← 仍刷新
  [FINAL #1] 你好我是 MOSS, 很高兴见到你
  [partial] 今天天气不错
  [HEALTH-CHANGE] ok → device_down (last_error: no PCM for 10.3s)
  [HEALTH-CHANGE] device_down → ok                                  ← 重连恢复

  按 Enter →
  [drain] items=2  forgotten=0
    [1] FINAL '你好我是 MOSS, 很高兴见到你'  ts=...
    [2] FINAL '今天天气不错'                ts=...
  [health] status=ok pending_partial=None ...

  ^C
  [stop] listener.stop() ...
  [summary] partial cb=156  sentence cb=2  drain=2  force drain=0

读完 docstring 还看不懂请回去读 runtime/README.md 和 listener.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession, patch_stdout

# 加载 .env (找 cwd 上各级 + .moss_ws/.env)
try:
    import dotenv

    _ws_env = None
    for p in [Path.cwd(), *Path.cwd().parents]:
        candidate = p / ".moss_ws" / ".env"
        if candidate.exists():
            _ws_env = candidate
            break
    if _ws_env is not None:
        dotenv.load_dotenv(_ws_env, override=False)
    else:
        dotenv.load_dotenv()  # 默认搜索
except ImportError:
    pass

from ghoshell_moss_contrib.unitree.g1.runtime import listener


# ── listener callbacks: 跑在 backend asyncio 线程, 不能阻塞 ──────────────

_partial_count = 0
_sentence_count = 0
_health_change_count = 0
_last_partial_text = ""


def _on_partial(u: listener.Utterance) -> None:
    """partial 高频刷新. patch_stdout 包裹下 print 不会破坏底部 prompt 输入框."""
    global _partial_count, _last_partial_text
    _partial_count += 1
    _last_partial_text = u.text
    # \r 覆盖刷新太冒险 (会跟 patch_stdout 冲突), 这里直接换行打印.
    # 高频可能刷屏, 这就是 partial listener 的真实代价 — 实际 channel 不会注册它.
    # 但为了让验证脚本看清流式过程, 我们容忍.
    print(f"[partial #{_partial_count:3d}] {u.text!r}")


def _on_sentence(u: listener.Utterance) -> None:
    global _sentence_count
    _sentence_count += 1
    tag = "FORCED" if u.forced else "FINAL"
    print(f"[{tag} #{_sentence_count}] {u.text!r}  ts={u.received_at:.2f}")


def _on_health_change(h: listener.ListenerHealth) -> None:
    global _health_change_count
    _health_change_count += 1
    detail = ""
    if h.last_error_msg:
        detail = f"  last_error={h.last_error_msg!r}"
    print(f"[HEALTH] → {h.status}{detail}")


# ── 主循环 ───────────────────────────────────────────────────────────────

def _format_batch(batch: listener.UtteranceBatch) -> str:
    lines = [f"[drain] items={len(batch.items)} forgotten={batch.forgotten}"]
    for i, u in enumerate(batch.items, 1):
        tag = "FORCED" if u.forced else ("FINAL" if u.is_final else "partial")
        lines.append(f"  [{i}] {tag:6s} {u.text!r}  ts={u.received_at:.2f}")
    return "\n".join(lines)


def _format_health(h: listener.ListenerHealth) -> str:
    # 一行精简版 + 多行完整版
    summary = (
        f"status={h.status}  device={h.device_name!r}  "
        f"sr_capture={h.sample_rate_capture}  sr_asr={h.sample_rate_asr}  "
        f"pending_partial={h.pending_partial!r}  "
        f"pending_drain={h.utterances_pending_drain}  "
        f"errors=(bt_lost={h.bt_lost_count} ws={h.ws_error_count} other={h.error_count})"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(prog="listener_sen_dialog", description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config",
        default=None,
        help="配置文件路径. 默认 ~/.moss_g1_listener.json",
    )
    args = ap.parse_args()

    # 配置文件位置. None 让 listener 走它自己的默认.
    config_path = Path(args.config).expanduser() if args.config else None

    # env 检查 (非阻塞警告)
    import os
    missing_env = [
        k for k in ("VOLCENGINE_BM_ASR_APPID", "VOLCENGINE_BM_ASR_TOKEN")
        if not os.environ.get(k)
    ]
    if missing_env:
        print(
            f"[warn] 缺环境变量: {', '.join(missing_env)}. "
            "ws 会失败, 但 capture 可独立验证."
        )
    print()

    # 注册 callback (在 start 之前注册无害, listener 会在 backend 起来后用)
    h_partial = listener.register_partial_listener(_on_partial)
    h_sentence = listener.register_sentence_listener(_on_sentence)
    h_health = listener.register_health_change_listener(_on_health_change)

    print(f"[bootstrap] listener.start(config_path={config_path}) ...")
    listener.start(config_path=config_path)

    # 立刻打一次 health, 让人知道初始状态
    time.sleep(0.3)
    h = listener.health()
    print(f"[bootstrap] health: {_format_health(h)}")
    print()

    print("=" * 72)
    print(" 对着耳机说话, 看 [partial] 流式刷新 → [FINAL] 服务端 VAD 判停.")
    print(" Enter   = drain (拿 finalized, partial 保留)")
    print(" f+Enter = drain(force_finalize_partial=True)  ← 打断 VAD, 拿当前 partial")
    print(" h+Enter = print full health snapshot")
    print(" Ctrl+C  = stop + 摘要")
    print("=" * 72)
    print()

    session: PromptSession = PromptSession()
    drain_count = 0
    force_drain_count = 0

    try:
        with patch_stdout.patch_stdout(raw=True):
            while True:
                try:
                    raw = session.prompt(">>> ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                cmd = raw.strip().lower()

                if cmd == "f":
                    force_drain_count += 1
                    batch = listener.drain(force_finalize_partial=True)
                    print(_format_batch(batch))
                elif cmd == "h":
                    h = listener.health()
                    print("[health snapshot]")
                    print(json.dumps(h.model_dump(), indent=2, ensure_ascii=False))
                elif cmd == "" or cmd is None:
                    drain_count += 1
                    batch = listener.drain()
                    print(_format_batch(batch))
                else:
                    print(f"[?] 未知命令 {cmd!r}. 用空 / f / h.")
    finally:
        print("\n[stop] listener.stop() ...")
        listener.unregister_listener(h_partial)
        listener.unregister_listener(h_sentence)
        listener.unregister_listener(h_health)
        listener.stop()
        print()
        print("=" * 72)
        print(
            f" 摘要: partial cb={_partial_count}  sentence cb={_sentence_count}  "
            f"health-change cb={_health_change_count}  "
            f"drain={drain_count}  force drain={force_drain_count}"
        )
        print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
