"""tts_integrity_probe — TTS → player 音频一致性探针 (跑完即退).

诊断「乱续 / 重音 / 断续」在哪一层. 两条路径:

- 默认 (直连): ``batch.items()`` → ``player.add()``, 与 TTS 源逐字节对比.
- ``--speech`` (全链路): ``speech.new_segment()`` → ``stream.play()``, 与 ``speak``
  命令同一条 Speech 管线 (只报播放样本, 拿不到独立源).

用法:

    moss --mode system_test nodes run .moss/system_test_nodes/tts_integrity_probe/ -- "一句话"
    moss --mode system_test nodes run .moss/system_test_nodes/tts_integrity_probe/ -- --speech "一句话"
"""

import asyncio
import sys
import time

import numpy as np

from ghoshell_moss.contracts.speech import AudioFormat, Speech, StreamAudioPlayer, TTS, TTSSpeech
from ghoshell_moss.core.blueprint.matrix import Matrix

_DEFAULT_TEXT = "一二三四五六七八九十，这是一句播放一致性测试。"


def _parse_argv() -> tuple[bool, str]:
    args = sys.argv[1:]
    use_speech = "--speech" in args
    args = [a for a in args if a != "--speech"]
    sentence = args[0] if args and args[0] else _DEFAULT_TEXT
    return use_speech, sentence


async def main(matrix: Matrix):
    use_speech, text = _parse_argv()
    con = matrix.container
    player = con.get(StreamAudioPlayer)
    if player is None:
        print("[fatal] StreamAudioPlayer 未注册", flush=True)
        return

    played: list[bytes] = []
    durations: list[float] = []

    def _on_sample(sample) -> None:
        played.append(sample.pcm)
        durations.append(sample.duration)

    unsub = player.observe(_on_sample)

    source_chunks: list[np.ndarray] | None = None
    info = None

    try:
        if use_speech:
            speech = con.get(Speech)
            if not isinstance(speech, TTSSpeech):
                print(f"[fatal] Speech 不是 TTSSpeech ({type(speech).__name__})", flush=True)
                return
            info = speech.tts().get_info()
            await speech.start()
            stream = speech.new_segment()
            stream.feed(text, complete=True)
            await stream.play()
        else:
            tts = con.get(TTS)
            if tts is None:
                print("[fatal] TTS 未注册", flush=True)
                return
            info = tts.get_info()
            audio_type = (
                AudioFormat(info.audio_format) if isinstance(info.audio_format, str) else info.audio_format
            )
            await tts.start()
            await player.start()
            batch = tts.new_batch(batch_id=f"integrity-{int(time.monotonic() * 1e9)}")
            batch.feed(text)
            batch.commit()
            await batch.start()
            source_chunks = []
            stream_id = f"integrity-{int(time.time())}"
            fid = [0]
            async for item in batch.items():
                audio = np.asarray(item["audio"])
                source_chunks.append(audio)
                player.add(
                    audio,
                    audio_type=audio_type,
                    rate=info.sample_rate,
                    channels=info.channels,
                    stream_id=stream_id,
                    fragment_id=str(fid[0]),
                )
                fid[0] += 1
            await player.wait_play_done()
    finally:
        unsub()

    played_bytes = b"".join(played)
    rate = info.sample_rate if info is not None else 0
    ms = [round(d * 1000, 1) for d in durations]
    print(
        f"[played] {len(played)} samples, {len(played_bytes) / 2 / rate:.2f}s "
        f"@ {rate}Hz",
        flush=True,
    )
    print(f"[chunks] {len(ms)} 块, 各块时长(ms): {ms}", flush=True)
    print(f"[chunks] 前 5 块: {ms[:5]}  后 3 块: {ms[-3:]}", flush=True)

    if source_chunks is None:
        print("[result] (speech 路径, 无独立源, 仅看上面的 chunk 分布)", flush=True)
        return

    source_bytes = b"".join(c.tobytes() for c in source_chunks)
    if source_bytes == played_bytes:
        verdict = "CONSISTENT — played == source (byte-identical)"
    elif source_bytes and len(played_bytes) % len(source_bytes) == 0 and played_bytes == source_bytes * (len(played_bytes) // len(source_bytes)):
        verdict = f"REPEATED ×{len(played_bytes) // len(source_bytes)} — 同一份音频播了多遍 (重音)"
    elif len(played_bytes) == len(source_bytes):
        verdict = "REORDERED — 长度相同但内容错位 (乱序)"
    else:
        verdict = f"LENGTH+CONTENT — byte diff = {len(played_bytes) - len(source_bytes)}"
    print(f"[result] {verdict}", flush=True)

    if source_bytes != played_bytes:
        n = min(len(source_bytes), len(played_bytes))
        if n:
            s = np.frombuffer(source_bytes[:n], dtype=np.int16)
            p = np.frombuffer(played_bytes[:n], dtype=np.int16)
            bad = np.flatnonzero(s != p)
            if bad.size:
                idx = int(bad[0])
                print(f"[result] first divergence at sample {idx} (≈{idx / rate:.2f}s)", flush=True)
            else:
                print("[result] common prefix identical; divergence is length-only", flush=True)

    await player.close()


if __name__ == "__main__":
    Matrix.discover().run(main)
