"""speak command — TTS synthesis to player playback, with optional WAV save.

探测 speak 协议:
  (2) TTS 片段 — text + 音频, 以 stream_id/fragment_id 为维度
  (3) 播放完成样本 — 真正"说了"的依据, fragment_id 可对齐回文本

优先走 Speech (TTSSpeech) 完整管线; --save 或 Speech 不可用时走直接 TTS + Player.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.audio.codec import _write_wav
from ghoshell_moss.cli.audio.render import (
    _render_from_queue,
    _report_playback_sample,
    _report_spectrogram,
)
from ghoshell_moss.cli.utils import is_ai_mode, print_error, print_info, print_success, print_warning
from ghoshell_moss.contracts.speech import AudioFormat, Speech, StreamAudioPlayer, TTS, TTSSpeech
from ghoshell_moss.core.blueprint.matrix import Matrix


@audio_app.command("speak")
def speak(
    text: str = typer.Argument(..., help="Text to synthesize and play."),
    tone: Optional[str] = typer.Option(None, "--tone", "-t", help="TTS voice/tone preset."),
    save: Optional[Path] = typer.Option(None, "--save", "-o", help="Save synthesized audio to a WAV file."),
) -> None:
    """Synthesize text to speech and play through audio output.

    Prefers Speech (TTSSpeech) provider to test the full TTS -> Player wiring.
    Falls back to direct TTS + Player pipeline when Speech is not available
    or when --save is given (which needs direct access to audio chunks).
    """
    matrix = Matrix.new("audio_speak", category="cli")
    result = matrix.run(lambda m: _async_speak(m, text=text, tone=tone, save=save))
    if result is None:
        return
    source, total, observed, interrupted, had_realtime = result
    if interrupted:
        print_warning("interrupted — playback stopped")
    else:
        snippet = text[:50] + ("..." if len(text) > 50 else "")
        print_success(f"speak finished: \"{snippet}\" — {source} {total:.2f}s")
    if had_realtime:
        return
    if is_ai_mode():
        _report_playback_sample(observed, total)
    else:
        _report_spectrogram(observed, total)


async def _async_speak(matrix, *, text: str, tone: Optional[str], save: Optional[Path]):
    """Dispatch to Speech path (preferred) or direct TTS + Player path."""
    con = matrix.container

    player = con.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered — run `moss audio contracts` to check.")
        return None

    speech = con.get(Speech)
    tts = con.get(TTS)

    # --save needs direct access to audio chunks; bypass Speech
    if save is not None or not isinstance(speech, TTSSpeech):
        if tts is None:
            print_error("TTS not registered — cannot synthesize speech.")
            return None
        return await _speak_direct(matrix, tts, player, text, tone, save)

    return await _speak_via_speech(matrix, speech, text, tone)


async def _speak_via_speech(matrix, speech, text: str, tone: Optional[str]):
    """Full TTS -> Player pipeline via Speech (TTSSpeech) wiring."""
    tts = speech.tts()
    player = speech.player()
    tts_info = tts.get_info()

    if tone:
        tts.use_tone(tone)

    print_info(
        f"TTS: {type(tts).__name__}  voice: {tts.current_tone()}  "
        f"rate: {tts_info.sample_rate}Hz  channels: {tts_info.channels}"
    )

    await speech.start()

    stream = speech.new_stream()
    stream.feed(text, complete=True)

    interrupted = False
    had_realtime = False
    observed: list = []
    _total = [0.0]

    def _collect(sample):
        observed.append(sample)
        _total[0] += sample.duration

    try:
        if not is_ai_mode():
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                _collect(sample)
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)
            try:
                say_task = asyncio.create_task(stream.say())
                render_task = asyncio.create_task(_render_from_queue(frame_q))
                done, pending = await asyncio.wait(
                    [say_task, render_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if say_task in done:
                    render_task.cancel()
                    had_realtime = True
                    if (exc := say_task.exception()) is not None:
                        raise exc
                else:
                    await say_task
            finally:
                unsub()
        else:
            unsub = player.observe(_collect)
            try:
                await stream.say()
            finally:
                unsub()
    except asyncio.CancelledError:
        interrupted = True
        await player.clear()
    finally:
        await speech.close()

    return f"speech:{type(tts).__name__}", _total[0], observed, interrupted, had_realtime


async def _speak_direct(matrix, tts, player, text: str, tone: Optional[str], save: Optional[Path]):
    """Direct TTS -> Player pipeline, with optional WAV save."""
    tts_info = tts.get_info()

    if tone:
        tts.use_tone(tone)

    audio_format = (
        AudioFormat(tts_info.audio_format)
        if isinstance(tts_info.audio_format, str)
        else tts_info.audio_format
    )

    print_info(
        f"TTS: {type(tts).__name__}  voice: {tts.current_tone()}  "
        f"rate: {tts_info.sample_rate}Hz  channels: {tts_info.channels}"
    )

    await tts.start()
    await player.start()

    batch = tts.new_batch(batch_id=f"cli-speak-{int(time.monotonic() * 1e9)}")
    batch.feed(text)
    batch.commit()
    await batch.start()

    stream_id = f"cli-speak-{int(time.time())}"
    all_audio: list = [] if save else None
    _total = [0.0]
    _fragment_id = [0]

    interrupted = False
    had_realtime = False
    observed: list = []

    def _collect(sample):
        observed.append(sample)

    try:
        if not is_ai_mode():
            loop = asyncio.get_running_loop()
            frame_q: asyncio.Queue = asyncio.Queue()

            def _on_sample(sample):
                _collect(sample)
                loop.call_soon_threadsafe(frame_q.put_nowait, sample)

            unsub = player.observe(_on_sample)

            async def _feed():
                async for item in batch.items():
                    audio = item["audio"]
                    player.add(
                        audio,
                        audio_type=audio_format,
                        rate=tts_info.sample_rate,
                        channels=tts_info.channels,
                        stream_id=stream_id,
                        fragment_id=str(_fragment_id[0]),
                    )
                    if all_audio is not None:
                        all_audio.append(audio.copy())
                    _fragment_id[0] += 1
                await player.wait_play_done()
                _total[0] = sum(s.duration for s in observed)

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
            unsub = player.observe(_collect)
            try:
                async for item in batch.items():
                    audio = item["audio"]
                    player.add(
                        audio,
                        audio_type=audio_format,
                        rate=tts_info.sample_rate,
                        channels=tts_info.channels,
                        stream_id=stream_id,
                        fragment_id=str(_fragment_id[0]),
                    )
                    if all_audio is not None:
                        all_audio.append(audio.copy())
                    _fragment_id[0] += 1
                await player.wait_play_done()
                _total[0] = sum(s.duration for s in observed)
            finally:
                unsub()
    except asyncio.CancelledError:
        interrupted = True
        await player.clear()
    finally:
        await tts.close()
        await player.close()

    if all_audio is not None and save:
        combined = np.concatenate(all_audio)
        _write_wav(save, combined, tts_info.sample_rate, tts_info.channels)
        print_success(f"saved {len(combined) / tts_info.sample_rate:.2f}s audio to {save}")

    return f"tts:{type(tts).__name__}", _total[0], observed, interrupted, had_realtime
