"""PlaybackSample observation display — spectrum, waveform, realtime frame rendering.

render 是"观测层"——把 PlaybackSample (协议 (3) 播放完成样本) 渲染给人或模型看.
消费方 (play/speak/echo/capture) 各自决定观测粒度, 渲染逻辑收拢在此.
"""

from __future__ import annotations

import asyncio
import sys
import time

import numpy as np

from ghoshell_moss.cli.utils import echo, is_ai_mode, print_info, print_simple_panel
from ghoshell_moss.topics.audio import AudioPlaybackTopic


def _render_frame(topic, first: bool = False) -> bool:
    """Render spectrum as horizontal bars — one row per frequency bin.

    Each row: [freq_label] [bar] dB_value.
    X axis = intensity (dB), Y axis = frequency (Hz, low→high top→bottom).
    """
    bins = topic.spectrum_bins
    if not bins:
        return first

    n_bins = len(bins)
    nyquist = topic.sample_rate / 2 if topic.sample_rate else 22050
    bin_hz = nyquist / n_bins  # Hz per bin

    bar_width = 40
    n_rows = n_bins

    if not first:
        sys.stdout.write(f"\033[{n_rows + 1}F")

    for i, db in enumerate(bins):
        freq = i * bin_hz
        if freq >= 1000:
            label = f"{freq / 1000:.1f}k".rjust(5)
        else:
            label = f"{freq:.0f}Hz".rjust(5)
        width = int((max(-60.0, min(0.0, db)) + 60.0) / 60.0 * bar_width)
        width = max(1, min(bar_width, width))
        bar = "█" * width + "░" * (bar_width - width)
        sys.stdout.write(f"\r {label} {bar} {db:+.1f}dB\n")

    sys.stdout.write(f"\r       peak {topic.peak:.2f}  rms {topic.rms_db:+.1f}dB\n")
    sys.stdout.flush()
    return False


async def _render_from_queue(queue: asyncio.Queue, first_frame_timeout: float = 2.0) -> bool:
    """Pull PlaybackSample from local queue, compute spectrum, render in real-time.

    Observer now fires at playback rate (BaseAudioStreamPlayer._wait_consumed),
    so frames arrive spaced by chunk duration — no burst, no polling hack needed.
    """
    first = True
    started = time.monotonic()
    rendered = False

    while True:
        try:
            sample = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if not rendered and time.monotonic() - started > first_frame_timeout:
                return False
            continue

        rendered = True
        bins = _spectrum_bins(sample, n_bins=16)
        frame = AudioPlaybackTopic(
            sample_rate=sample.sample_rate,
            rms_db=sample.rms_db,
            peak=sample.peak,
            spectrum_bins=bins,
            n_spectrum_bins=16,
        )
        first = _render_frame(frame, first=first)


def _report_playback_sample(observed, total: float) -> None:
    """报告实际播放可感知样本. 单次 add 通常 1 帧; --ai 模式最多一行."""
    if not observed:
        if is_ai_mode():
            echo(f"played {total:.2f}s — no playback samples observed")
        else:
            print_info(f"played {total:.2f}s — no playback samples observed")
        return
    s = observed[0]
    pcm_sz = len(s.pcm) if s.pcm else 0
    if is_ai_mode():
        echo(
            f"played {total:.2f}s — sample: stream={s.stream_id or '-'} fragment={s.fragment_id or '-'} "
            f"rms={s.rms_db:.1f}dB peak={s.peak:.3f} pcm={pcm_sz}B @{s.sample_rate}Hz"
        )
        return
    from rich.text import Text

    t = Text()
    t.append(f"stream={s.stream_id or '-'}  fragment={s.fragment_id or '-'}\n", style="dim")
    t.append(f"duration={s.duration:.2f}s  rms={s.rms_db:.1f}dB  peak={s.peak:.3f}\n")
    bar_w = 28
    lvl = int((max(-60.0, min(0.0, s.rms_db)) + 60.0) / 60.0 * bar_w)
    t.append("rms  " + "█" * lvl + "░" * (bar_w - lvl) + "\n", style="yellow")
    t.append(f"pcm={pcm_sz}B @{s.sample_rate}Hz", style="dim")
    print_simple_panel(t, title="playback sample")


def _spectrum_bins(sample, n_bins: int = 10) -> list[float]:
    """从 PlaybackSample.pcm 做 FFT, 返回 n_bins 个频段能量 (dB).

    消费方自行选择 bin 数——10 是频谱谱面, 20 可画波浪线.
    """
    if not sample.pcm:
        return [-96.0] * n_bins
    pcm = np.frombuffer(sample.pcm, dtype=np.int16).astype(np.float64) / 32768.0
    fft = np.abs(np.fft.rfft(pcm))
    n_fft = len(fft)
    if n_fft < n_bins * 2:
        return [float(20.0 * np.log10(max(fft.mean(), 1e-10)))] * n_bins
    bins = []
    for i in range(n_bins):
        lo = int(i * n_fft / n_bins)
        hi = int((i + 1) * n_fft / n_bins)
        db = 20.0 * np.log10(max(float(fft[lo:hi].mean()), 1e-10))
        bins.append(round(db, 1))
    return bins


def _render_spectrogram(observed, n_bins: int = 10, max_rows: int = 40) -> str:
    """N 个频段的文本频谱谱面 — 每行一个片段, 堆叠即"跳跃的波浪线".

    返回一个 Text 对象, human 模式 rich Panel 输出; --ai 模式 echo 纯文本.
    """
    if not observed:
        return "(no samples)"
    bars = "▁▂▃▄▅▆▇█"
    rows = []
    for s in observed[-max_rows:]:
        bins = _spectrum_bins(s, n_bins=n_bins)
        row = "".join(bars[max(0, min(7, int((db + 60.0) / 60.0 * 7.99)))] for db in bins)
        rows.append(row)
    return "\n".join(rows)


def _report_spectrogram(observed, total: float) -> None:
    """渲染频谱谱面 — human 模式 rich 面板, --ai 模式纯文本行."""
    spectro = _render_spectrogram(observed, n_bins=10)
    if is_ai_mode():
        echo(spectro)
        if observed:
            dbs = [s.rms_db for s in observed]
            echo(f"fragments: {len(observed)}  rms range=[{min(dbs):.1f}, {max(dbs):.1f}] dB")
        return
    from rich.text import Text

    t = Text()
    t.append(f"fragments: {len(observed)}  ", style="dim")
    t.append(f"duration: {total:.2f}s\n")
    t.append(spectro, style="yellow")
    print_simple_panel(t, title="playback spectrogram")
