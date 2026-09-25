"""voice_echo_probe — 单进程 听+说 并发, 纯观测回声 (跑完即退).

AEC 已由 miniaudio factory 在两条 stream 都产出时自动装好 (见
``ghoshell_moss.host.audios.miniaudio_impl.factory``), 本 node 不再手动装线,
也不走 listener controller. 所以这里就是两个并发 task:

- 嘴说: TTS 播放一句话 (``_speak``).
- 耳朵听: capture 顺序消费, 逐帧打印 rms (``_listen``).

耳朵打印的是**消回声后**的信号 (capture 在算 meta 前就过了 AEC), 因此:

- 说话期间耳朵侧 rms 低 / SILENT = 回声被消掉;
- 说话期间耳朵侧 rms 跳高 = 回声漏进来了.

判据是时间轴上的对照: 静默基线 → ``[say]`` 标记 → 说话窗口 → ``[say done]``
标记 → 尾巴窗口. 说话窗口里耳朵若仍是 SILENT 就是好的.

**外放跑** — 要测的就是扬声器 → 麦克风这条路, 戴耳机测到的是"没回声".

用法:

    moss --mode system_test nodes run .moss/system_test_nodes/voice_echo_probe/ -- "要说的一句话"

句子可省, 缺省用内置测试句; 控制句子长度可观察 AEC 收敛期与稳态的不同.
设备选择走 node 自身的 dotenv (MOSS_AUDIO_CAPTURE_DEVICE).
"""

import asyncio
import sys
import time

from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.speech import PlaybackSample, Speech, TTSSpeech
from ghoshell_moss.core.blueprint.matrix import Matrix

#: 说话前留一段静默基线 — 确认说话前耳朵是 SILENT 的.
_SETTLE_SECONDS = 1.5

#: 说完之后观测多久 (覆盖回声尾巴), 再收尾.
_TAIL_SECONDS = 3.0

_DEFAULT_SENTENCE = "你好，这是一句测试。听一听，耳朵能不能听见我自己说话。"


def _parse_argv() -> str:
    """``-- "<sentence>"`` — 可省, 缺省用内置测试句."""
    args = sys.argv[1:]
    return args[0] if len(args) > 0 and args[0] else _DEFAULT_SENTENCE


async def main(matrix: Matrix):
    sentence = _parse_argv()
    con = matrix.container

    speech = con.get(Speech)
    if not isinstance(speech, TTSSpeech):
        print(f"[fatal] Speech 不是 TTSSpeech ({type(speech).__name__}), 无法出声", flush=True)
        return

    capture = con.get(AudioCaptureSource)
    if capture is None:
        print("[fatal] AudioCaptureSource 未注册", flush=True)
        return

    # 取到 player + capture 后, factory 已在两条 stream 齐时自动装好 AEC.
    await speech.start()
    await capture.start()
    player = speech.player()
    print(
        f"[boot] player={type(player).__name__} {player.sample_rate}Hz/{player.channels}ch  "
        f"capture={capture.device_explain()}",
        flush=True,
    )

    consumer = capture.new_sequential_consumer()

    async def _speak() -> None:
        samples: list[PlaybackSample] = []
        stream = speech.new_segment()
        stream.feed(sentence, complete=True)
        await stream.play(samples)
        played = sum(s.duration for s in samples)
        print(f"[say done] played {played:.2f}s over {len(samples)} samples", flush=True)

    async def _listen() -> None:
        t0 = time.monotonic()
        async with consumer:
            async for chunk in consumer:
                m = chunk.meta
                tag = "SILENT" if m.is_silent else "     "
                print(f"[ear] +{time.monotonic() - t0:6.2f}s rms={m.rms_db:6.1f}dB {tag}", flush=True)

    listen_task = asyncio.create_task(_listen())

    # 静默基线: 说话前耳朵应当是 SILENT.
    await asyncio.sleep(_SETTLE_SECONDS)

    print(f'[say] "{sentence}"', flush=True)
    speak_task = asyncio.create_task(_speak())
    try:
        await speak_task
    except Exception as exc:
        print(f"[say] failed: {exc}", flush=True)

    print(f"[observe] 再听 {_TAIL_SECONDS:.0f}s 收尾 (看回声尾巴) ...", flush=True)
    await asyncio.sleep(_TAIL_SECONDS)

    listen_task.cancel()
    speak_task.cancel()
    await asyncio.gather(listen_task, speak_task, return_exceptions=True)

    await capture.close()
    try:
        await speech.close()
    except BaseException:
        pass
    print("[done] exiting", flush=True)


if __name__ == "__main__":
    Matrix.discover().run(main)
