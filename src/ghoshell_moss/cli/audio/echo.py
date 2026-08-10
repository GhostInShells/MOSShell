"""echo command — capture then play back with real-time spectrum.

组合 capture 协议 (1) 音频聆听片段 与 speak 协议 (3) 播放完成样本:
  capture N 秒 → 片段切片喂 player → 实时频谱渲染.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import typer

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.audio.capture import _async_capture
from ghoshell_moss.cli.audio.codec import _fragments
from ghoshell_moss.cli.audio.render import _render_from_queue
from ghoshell_moss.cli.utils import is_ai_mode, print_error, print_success, print_warning
from ghoshell_moss.contracts.speech import AudioFormat, StreamAudioPlayer
from ghoshell_moss.core.blueprint.matrix import Matrix


@audio_app.command("echo")
def echo_cmd(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Capture duration in seconds."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern (empty for default)."),
) -> None:
    """Capture audio then play it back immediately with real-time spectrum."""
    matrix = Matrix.new("audio_echo", category="cli")
    result = matrix.run(lambda m: _async_echo(m, seconds=seconds, device=device))
    if result is None:
        return
    total, interrupted = result
    if interrupted:
        print_warning("echo interrupted")
    else:
        print_success(f"echo finished: {total:.2f}s")


async def _async_echo(matrix, *, seconds: float, device: Optional[str]):
    """Capture audio then play back through StreamAudioPlayer."""
    cap_result = await _async_capture(matrix, seconds=seconds, save=None, device=device)
    if cap_result is None:
        return None
    pcm, rate, cap_total, interrupted = cap_result

    if interrupted:
        return cap_total, True

    player = matrix.container.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered")
        return None

    await player.start()

    stream_id = f"cli-echo-{int(time.time())}"
    echo_interrupted = False

    try:
        if not is_ai_mode():
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)

            async def _feed():
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
                feed_task = asyncio.create_task(_feed())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [feed_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if feed_task in done:
                    render_task.cancel()
                    if (exc := feed_task.exception()) is not None:
                        raise exc
                else:
                    await feed_task
            finally:
                unsub()
        else:
            player.add(
                pcm,
                audio_type=AudioFormat.PCM_S16LE,
                rate=rate,
                stream_id=stream_id,
                fragment_id="0",
            )
            await player.wait_play_done()
    except asyncio.CancelledError:
        echo_interrupted = True
        await player.clear()
    finally:
        await player.close()

    return cap_total, echo_interrupted
