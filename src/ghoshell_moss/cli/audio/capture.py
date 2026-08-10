"""capture command — record N seconds via AudioCaptureSource, show spectrogram, optional WAV save.

探测 capture 协议 (1): 音频聆听片段. 采集边界由 AudioCaptureSource 自持,
CLI 经 sequential consumer 拉取原始 PCM 切片.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.audio.codec import _fragments, _write_wav
from ghoshell_moss.cli.audio.render import _render_spectrogram, _report_spectrogram
from ghoshell_moss.cli.utils import echo, is_ai_mode, print_error, print_success, print_warning
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.speech import PlaybackSample
from ghoshell_moss.core.blueprint.matrix import Matrix


@audio_app.command("capture")
def capture(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Capture duration in seconds."),
    save: Optional[Path] = typer.Option(None, "--save", "-o", help="Save captured audio to WAV file."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Capture device name pattern (empty for default)."),
) -> None:
    """Capture audio for N seconds, show waveform, optionally save to WAV."""
    matrix = Matrix.new("audio_capture", category="cli")
    result = matrix.run(lambda m: _async_capture(m, seconds=seconds, save=save, device=device))
    if result is None:
        return
    pcm, rate, total, interrupted = result
    if interrupted:
        print_warning("capture interrupted")
    else:
        print_success(f"captured {total:.2f}s @{rate}Hz")
    _report_capture_spectrogram(pcm, rate, total)


async def _async_capture(matrix, *, seconds: float, save: Optional[Path], device: Optional[str]):
    """Capture audio from the default input device."""
    con = matrix.container

    capture_source = con.get(AudioCaptureSource)
    if capture_source is None:
        print_error("AudioCaptureSource not registered")
        return None

    if device is not None:
        capture_source._config.device_pattern = device

    await capture_source.start()

    if "not started" in capture_source.device_explain():
        print_error("capture device not started — may be locked by another process")
        await capture_source.close()
        return None

    print_info(f"capturing {seconds}s from {capture_source.device_explain()}...")

    consumer = capture_source.new_sequential_consumer(max_queue_frames=256)
    await consumer.start()

    all_audio: list = []
    sample_rate = capture_source._config.sample_rate
    channels = capture_source._config.channels
    interrupted = False

    try:
        deadline = time.monotonic() + seconds
        async for chunk in consumer:
            all_audio.append(chunk.samples.copy())
            if time.monotonic() >= deadline:
                break
    except asyncio.CancelledError:
        interrupted = True
    finally:
        await consumer.close()
        await capture_source.close()

    if not all_audio:
        print_warning("no audio captured — is the device working?")
        return None

    combined = np.concatenate(all_audio)
    total = len(combined) / sample_rate

    if save:
        _write_wav(save, combined, sample_rate, channels)
        print_success(f"saved {total:.2f}s to {save}")

    return combined, sample_rate, total, interrupted


def _report_capture_spectrogram(pcm: np.ndarray, rate: int, total: float) -> None:
    """Show spectrogram of captured audio — reuse existing rendering."""
    frags = _fragments(pcm, rate, frag_ms=100)
    observed = []
    for frag in frags:
        f32 = frag.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(f32 ** 2)))
        rms_db = 20.0 * np.log10(max(rms, 1e-10))
        peak = float(np.max(np.abs(f32)))
        observed.append(PlaybackSample(
            pcm=frag.tobytes(),
            duration=len(frag) / rate,
            sample_rate=rate,
            rms_db=round(rms_db, 1),
            peak=round(peak, 3),
        ))
    if is_ai_mode():
        spectro = _render_spectrogram(observed, n_bins=10)
        echo(spectro)
        dbs = [s.rms_db for s in observed]
        echo(f"fragments: {len(observed)}  rms range=[{min(dbs):.1f}, {max(dbs):.1f}] dB")
    else:
        _report_spectrogram(observed, total)
