"""voice_echo_probe — 单进程 听+说 共存与回声实测 node (跑完即退, 纯观测).

一个进程里同时拉起两个器官, 启动 → 说一句 → 观测一个窗口 → **退出**. 不做常驻,
不做 singleton idle — 纯测试不持久运行.

回答两个问题:

1. **共存**: 麦克风 (capture) 与扬声器 (playback) 同进程起来会不会打架 —
   device 争用 / stream 冲突 / 生命周期. `[boot] both up` 出现即两者共存成立.
2. **回声**: 起来之后让嘴说一句, 看耳朵有没有听见自己说的话. 听见了就是回声,
   识别事件打印里的 Δ 就是回声延迟 (相对"说话起点").

耳朵全程 ``always`` 礼仪 (纯 segment_vad), 不让 LLM 进观测链路. 每个事件带
``seg=`` 后四位, 好分得清"切段"还是"只认一半".

**外放跑** — 要测的就是扬声器 → 麦克风这条路. 戴耳机测到的是"没回声".

用法:

    moss --mode system_test nodes run .moss/system_test_nodes/voice_echo_probe/ -- "要说的一句话"

句子可省, 缺省用内置测试句. 设备选择走 node 自身的 dotenv (MOSS_AUDIO_CAPTURE_DEVICE).
"""

import asyncio
import sys
import time
from typing import Optional

from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.speech import PlaybackSample, Speech, TTSSpeech
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.nodes.listener_node import assemble_controller

#: 两个器官都起来之后, 等这么久再说第一句话 — 留给设备/ASR 连接稳定.
_SETTLE_SECONDS = 3.0

#: 等 listening session 真正起来的上限.
_READY_TIMEOUT_SECONDS = 15.0

#: 说完之后观测多久 (覆盖 clause/tail 的回声尾巴), 然后退出.
_OBSERVE_SECONDS = 8.0

_DEFAULT_SENTENCE = "你好，这是一句测试。听一听，耳朵能不能听见我自己说话。"


def _parse_argv() -> str:
    """``-- "<sentence>"`` — 可省, 缺省用内置测试句."""
    args = sys.argv[1:]
    return args[0] if len(args) > 0 and args[0] else _DEFAULT_SENTENCE


class _SayClock:
    """说话时间轴 — 识别事件的 Δ 一律相对它算."""

    def __init__(self) -> None:
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

    def delta(self, now: float) -> str:
        if self.started_at is None:
            return "  --  "
        return f"+{now - self.started_at:5.2f}s"


async def _wait_listening(controller, timeout: float) -> bool:
    """轮询到 listening session 真的打开 (capture + asr 都活了)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if controller.snapshot().listening:
            return True
        await asyncio.sleep(0.1)
    return False


async def main(matrix: Matrix):
    sentence = _parse_argv()
    clock = _SayClock()
    log = matrix.logger

    print("[boot] listener: assembling (capture + asr) ...", flush=True)
    controller = await assemble_controller(matrix, emit_signals=False)

    speech = matrix.container.get(Speech)
    if not isinstance(speech, TTSSpeech):
        print(f"[boot] FATAL: Speech is {type(speech).__name__}, not TTSSpeech — 无法出声", flush=True)
        return

    print("[boot] speech: starting (player + tts) ...", flush=True)
    await speech.start()
    player = speech.player()
    print(
        f"[boot] speech: started — player={type(player).__name__} "
        f"{player.sample_rate}Hz/{player.channels}ch",
        flush=True,
    )

    capture = matrix.container.get(AudioCaptureSource)
    capture_explain = capture.device_explain() if capture is not None else "<not provided>"
    print(f"[boot] capture: {capture_explain}", flush=True)
    print("[boot] both up", flush=True)

    def _on_result(result: RecognitionEvent) -> None:
        now = time.monotonic()
        if result.phase is RecognitionPhase.CLAUSE and result.clause is not None:
            text = result.clause.text
        else:
            text = result.text
        sid = result.segment_id[-4:] if result.segment_id else "????"
        print(f"[{result.phase.value:<7} {clock.delta(now)} seg={sid}] {text}", flush=True)

    controller.on_recognition_result(_on_result)

    print("[boot] listener: opening listening session (always) ...", flush=True)
    controller.always(timeout=None)
    if await _wait_listening(controller, _READY_TIMEOUT_SECONDS):
        print("[boot] listener: listening", flush=True)
    else:
        print(
            f"[boot] WARN: listener 未在 {_READY_TIMEOUT_SECONDS}s 内 listening — "
            f"继续, 但耳朵可能是哑的",
            flush=True,
        )

    await asyncio.sleep(_SETTLE_SECONDS)

    clock.started_at = time.monotonic()
    print(f'[say] "{sentence}"', flush=True)
    samples: list[PlaybackSample] = []
    stream = speech.new_segment()
    stream.feed(sentence, complete=True)
    try:
        await stream.play(samples)
    except Exception as exc:
        log.exception("say failed: %s", exc)
    finally:
        clock.ended_at = time.monotonic()

    played = sum(sample.duration for sample in samples)
    print(
        f"[say done] played {played:.2f}s over {len(samples)} samples "
        f"(播放窗口 = Δ 0.00s → {clock.delta(clock.ended_at)})",
        flush=True,
    )

    # 观测窗口: 覆盖 clause/tail 的回声尾巴, 然后退出.
    print(f"[observe] 再听 {_OBSERVE_SECONDS:.0f}s 收尾, 之后自动退出", flush=True)
    await asyncio.sleep(_OBSERVE_SECONDS)
    print("[done] observation complete — exiting", flush=True)

    controller.stop()
    try:
        await speech.close()
    except BaseException:
        pass


if __name__ == "__main__":
    Matrix.discover().run(main)
