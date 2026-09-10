"""asr command — capture → ASR streaming recognition → live transcript.

探测 capture 协议 (2): 音频片段 + ASR 结果.
云端 VAD 决定分句边界, CLI 不做 VAD — 只观察协议行为.
新契约下 recognize() 是连续 loop: 一次调用消费整条音频流, 逐相位产出
partial (中间) / clause (分句) / tail (尾包).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.audio.codec import _write_wav
from ghoshell_moss.cli.utils import echo, is_ai_mode, print_error, print_info, print_success, print_warning
from ghoshell_moss.contracts.asr import ASR, RecognitionPhase
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.core.blueprint.matrix import Matrix


@audio_app.command("asr")
def asr_cmd(
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="Session timeout in seconds. Auto-stops on silence after speech."),
    save: Optional[Path] = typer.Option(None, "--save", "-o", help="Save captured audio to WAV file."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern."),
    json_mode: bool = typer.Option(False, "--json", help="Output RecognitionResult records as JSON lines."),
) -> None:
    """Capture audio and stream through ASR — live transcript with cloud VAD clause boundaries."""
    matrix = Matrix.new("audio_asr", category="cli")
    result = matrix.run(lambda m: _async_asr(m, timeout=timeout, save=save, device=device, json_mode=json_mode))
    if result is None:
        return
    total_duration, clause_count, interrupted = result
    if interrupted:
        print_warning("session interrupted")
    else:
        print_success(f"session done: {total_duration:.1f}s, {clause_count} clauses")


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
            f"device={capture_source.device_explain()}  "
            f"capture={sample_rate}Hz  asr={target_rate}Hz  timeout={timeout}s"
        )
        echo("speak now — Ctrl+C to stop\n")

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

    async def _bridge(consumer):
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

    interrupted = False
    clause_count = 0
    session_start = time.monotonic()

    async def _audio_gen():
        """Yield int16 samples from the bridge queue until deadline."""
        deadline = session_start + timeout
        while time.monotonic() < deadline:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                yield chunk
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    consumer = capture_source.new_sequential_consumer(max_queue_frames=256)
    try:
        async with consumer:
            bridge_task = asyncio.create_task(_bridge(consumer))
            try:
                recognition = asr.recognize(_audio_gen())
                async for result in recognition:
                    if result.error:
                        _commit_line(f"[错误] {result.error}")
                        print_error(result.error)
                        break
                    elapsed = time.monotonic() - session_start
                    if result.phase == RecognitionPhase.CLAUSE:
                        clause_count += 1
                    if json_mode:
                        echo(json.dumps({
                            "stream_id": result.stream_id,
                            "segment_id": result.segment_id,
                            "text": result.text,
                            "phase": result.phase.value,
                            "start_ms": result.start_ms,
                            "end_ms": result.end_ms,
                            "elapsed": round(elapsed, 3),
                            "error": result.error or None,
                        }, ensure_ascii=False))
                    elif is_ai_mode():
                        if result.phase == RecognitionPhase.CLAUSE:
                            echo(result.text)
                            echo("---")
                    else:
                        if result.phase == RecognitionPhase.PARTIAL:
                            _live_write(result.text)
                        elif result.phase == RecognitionPhase.CLAUSE:
                            _commit_line(result.text)
                            echo("---")
                        elif result.phase == RecognitionPhase.TAIL:
                            break
            finally:
                bridge_task.cancel()
                try:
                    await bridge_task
                except asyncio.CancelledError:
                    pass
    except asyncio.CancelledError:
        interrupted = True
    finally:
        await capture_source.close()
        await asr.close()

    total_duration = time.monotonic() - session_start

    if all_audio and save:
        combined = np.concatenate(all_audio)
        _write_wav(save, combined, sample_rate, channels)
        print_success(f"saved {len(combined) / sample_rate:.2f}s audio to {save}")

    return total_duration, clause_count, interrupted


def _live_write(text: str) -> None:
    """Update the current terminal line in-place with partial ASR text."""
    sys.stdout.write(f"\r\033[K  {text}")
    sys.stdout.flush()


def _commit_line(text: str) -> None:
    """Clear the live-update line and print the final text."""
    sys.stdout.write(f"\r\033[K  {text}\n")
    sys.stdout.flush()
