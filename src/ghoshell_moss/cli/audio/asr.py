"""asr command — capture → ASR streaming recognition → live transcript.

探测 capture 协议 (2): 音频片段 + ASR 结果.
云端 VAD 决定话语边界, CLI 不做 VAD — 只观察协议行为.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.audio.codec import _write_wav
from ghoshell_moss.cli.utils import echo, is_ai_mode, print_error, print_info, print_success, print_warning
from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.listener._asr_helpers import iter_with_silence_timeout


@audio_app.command("asr")
def asr_cmd(
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="Session timeout in seconds. Auto-stops on silence after speech."),
    save: Optional[Path] = typer.Option(None, "--save", "-o", help="Save captured audio to WAV file."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern."),
    json_mode: bool = typer.Option(False, "--json", help="Output ASRResult records as JSON lines."),
) -> None:
    """Capture audio and stream through ASR — live transcript with cloud VAD turn boundaries."""
    matrix = Matrix.new("audio_asr", category="cli")
    result = matrix.run(lambda m: _async_asr(m, timeout=timeout, save=save, device=device, json_mode=json_mode))
    if result is None:
        return
    total_duration, turn_count, interrupted = result
    if interrupted:
        print_warning("session interrupted")
    else:
        print_success(f"session done: {total_duration:.1f}s, {turn_count} turns")


async def _async_asr(matrix, *, timeout: float, save: Optional[Path], device: Optional[str], json_mode: bool):
    con = matrix.container

    asr = con.get(ASR)
    if asr is None:
        print_error("ASR not registered — run `moss audio contracts` to check.")
        return None

    capture_source = con.get(AudioCaptureSource)
    if capture_source is None:
        print_error("AudioCaptureSource not registered")
        return None

    asr_info = asr.get_info()

    if device is not None:
        capture_source._config.device_pattern = device

    await capture_source.start()

    if "not started" in capture_source.device_explain():
        print_error("capture device not started — may be locked by another process")
        await capture_source.close()
        return None

    sample_rate = capture_source._config.sample_rate
    channels = capture_source._config.channels
    target_rate = asr_info.sample_rate

    if not json_mode and not is_ai_mode():
        print_info(
            f"device={capture_source.device_explain()}  model={asr_info.model}  "
            f"capture={sample_rate}Hz  asr={target_rate}Hz  timeout={timeout}s"
        )
        echo("speak now — Ctrl+C to stop\n")

    consumer = capture_source.new_sequential_consumer(max_queue_frames=256)
    await consumer.start()

    # Bridge: consumer → asyncio.Queue (background task continuously fills queue)
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    all_audio: list = []

    def _resample(audio_data: np.ndarray, origin_rate: int, target_rate: int) -> np.ndarray:
        if origin_rate == target_rate:
            return audio_data
        target_len = int(len(audio_data) * target_rate / origin_rate)
        x_orig = np.arange(len(audio_data))
        x_target = np.linspace(0, len(audio_data) - 1, target_len)
        return np.interp(x_target, x_orig, audio_data).astype(np.int16)

    async def _bridge():
        try:
            async for chunk in consumer:
                samples = chunk.samples.copy()
                all_audio.append(samples)
                pcm = samples.ravel().astype(np.int16)
                if len(pcm) == 0:
                    continue
                if sample_rate != target_rate:
                    pcm = _resample(pcm, sample_rate, target_rate)
                await audio_queue.put(pcm)
        except asyncio.CancelledError:
            pass

    bridge_task = asyncio.create_task(_bridge())
    interrupted = False
    turn_count = 0
    session_start = time.monotonic()
    logger = logging.getLogger("moss.audio.asr")

    async def _audio_gen():
        """Yield int16 samples from the bridge queue until deadline.

        Uses a short timeout on queue.get() so abandoned generators from
        previous turns clean up their waiter promptly.
        """
        deadline = session_start + timeout
        while time.monotonic() < deadline:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                yield chunk
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    try:
        deadline = session_start + timeout
        while time.monotonic() < deadline:
            # Drain stale audio between turns
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            turn_count += 1
            try:
                async for result in iter_with_silence_timeout(
                    asr.recognize(_audio_gen()),
                    logger,
                    patience=5.0,
                ):
                    if result.error:
                        _commit_line(f"[错误] {result.error}")
                        print_error(result.error)
                        break
                    if not result.text:
                        continue
                    elapsed = time.monotonic() - session_start
                    if json_mode:
                        echo(json.dumps({
                            "text": result.text,
                            "is_final": result.is_final,
                            "elapsed": round(elapsed, 3),
                            "turn": turn_count,
                            "error": result.error or None,
                        }, ensure_ascii=False))
                    elif is_ai_mode():
                        if result.is_final:
                            echo(result.text)
                            echo("---")
                    else:
                        if result.is_final:
                            _commit_line(result.text)
                            echo("---")
                        else:
                            _live_write(result.text)
                    if result.is_final:
                        break
            except asyncio.CancelledError:
                interrupted = True
                break

    except asyncio.CancelledError:
        interrupted = True
    finally:
        bridge_task.cancel()
        try:
            await bridge_task
        except asyncio.CancelledError:
            pass
        await consumer.close()
        await capture_source.close()
        await asr.close()

    total_duration = time.monotonic() - session_start

    if all_audio and save:
        combined = np.concatenate(all_audio)
        _write_wav(save, combined, sample_rate, channels)
        print_success(f"saved {len(combined) / sample_rate:.2f}s audio to {save}")

    return total_duration, turn_count, interrupted


def _live_write(text: str) -> None:
    """Update the current terminal line in-place with partial ASR text."""
    sys.stdout.write(f"\r\033[K  {text}")
    sys.stdout.flush()


def _commit_line(text: str) -> None:
    """Clear the live-update line and print the final text."""
    sys.stdout.write(f"\r\033[K  {text}\n")
    sys.stdout.flush()
