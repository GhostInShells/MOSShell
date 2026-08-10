"""play command — pure output sanity (tune/wav), optional realtime spectrum.

无协议探测的纯输出体感: 播放标准 tune 或 WAV, 验证体感, 支持中断与波形.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.audio.codec import _fragments, _read_wav, _synthesize_tune
from ghoshell_moss.cli.audio.render import (
    _render_from_queue,
    _report_playback_sample,
    _report_spectrogram,
)
from ghoshell_moss.cli.utils import is_ai_mode, print_error, print_success, print_warning
from ghoshell_moss.contracts.speech import AudioFormat, StreamAudioPlayer
from ghoshell_moss.core.blueprint.matrix import Matrix


@audio_app.command("play")
def play(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Standard tune duration in seconds (ignored when --file is given)."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Play a WAV file instead of the standard tune."),
    pub_topic: bool = typer.Option(False, "--pub-topic", hidden=True, help="Publish AudioPlaybackTopic via Zenoh for remote consumers."),
) -> None:
    """Play a standard tune (or a WAV file) to test audible playback feel. Ctrl+C to interrupt.

    Human 模式: 播放中实时频谱柱状图. --ai 模式: 单帧样本摘要.
    """
    matrix = Matrix.new("audio_play", category="cli")
    result = matrix.run(lambda m: _async_play(m, seconds=seconds, file=file, pub_topic=pub_topic))

    if result is None:
        return
    source, total, observed, interrupted, had_realtime = result
    if interrupted:
        print_warning("interrupted — playback stopped")
    else:
        print_success(f"playback finished: {source} {total:.2f}s")
    if had_realtime:
        return  # 已在播放期间实时渲染
    if is_ai_mode():
        _report_playback_sample(observed, total)
    else:
        _report_spectrogram(observed, total)


async def _async_play(matrix, *, seconds: float, file: Optional[Path], pub_topic: bool = False):
    """收集播放数据; 本地 queue 桥接实时渲染, --pub-topic 时额外走 Zenoh 广播."""
    player = matrix.container.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered — run `moss audio contracts` to see what's available.")
        return None

    if file is not None:
        if not file.exists():
            print_error(f"file not found: {file}")
            return None
        pcm, rate = _read_wav(file)
        source = str(file)
    else:
        rate = 44100
        pcm = _synthesize_tune(seconds, rate)
        source = "tune"
    if len(pcm) == 0:
        print_warning("no audio to play")
        return None
    total = len(pcm) / rate

    await player.start()

    if pub_topic:
        player.enable_playback_topic()

    stream_id = f"cli-{int(time.time())}"
    observed: list = []
    interrupted = False
    had_realtime = False

    async def _feed():
        if is_ai_mode():
            player.add(
                pcm,
                audio_type=AudioFormat.PCM_S16LE,
                rate=rate,
                stream_id=stream_id,
                fragment_id="0",
            )
        else:
            for i, frag in enumerate(_fragments(pcm, rate)):
                player.add(
                    frag,
                    audio_type=AudioFormat.PCM_S16LE,
                    rate=rate,
                    stream_id=stream_id,
                    fragment_id=str(i),
                )
        await player.wait_play_done()

    try:
        if not is_ai_mode():
            # 本地桥: observer → call_soon_threadsafe → asyncio.Queue → render task
            # observer 现在以播放速率触发 (_wait_consumed), 帧按 chunk 间隔到达
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)
            try:
                feed_task = asyncio.create_task(_feed())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [feed_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if feed_task in done:
                    render_task.cancel()
                    had_realtime = True
                    if (exc := feed_task.exception()) is not None:
                        raise exc
                else:
                    await feed_task
            finally:
                unsub()
        else:
            await _feed()
    except asyncio.CancelledError:
        interrupted = True
        await player.clear()
    finally:
        await player.close()

    return source, total, observed, interrupted, had_realtime
