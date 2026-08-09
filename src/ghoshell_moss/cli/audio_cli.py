"""Audio CLI — capability probing for audio capture, playback, TTS, ASR.

CLI 以 cli 身份声明为 matrix node (Matrix.new), 从容器查询音频抽象的注册 provider.
只调 get_provider 反映注册可用性, 不实例化实现 — 不触发重 import, 无副作用.
仅构造容器, 不 join 网络.
"""

from __future__ import annotations

import asyncio
import time
import wave as _wave
from pathlib import Path
from typing import Optional, Type

import numpy as np
import typer

from ghoshell_container import INSTANCE

from ghoshell_moss.cli.utils import (
    console,
    echo,
    is_ai_mode,
    print_error,
    print_info,
    print_simple_table,
    print_success,
    print_warning,
)
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.speech import AudioFormat, Speech, StreamAudioPlayer, TTS

audio_app = typer.Typer(
    help="Audio capability probing — capture, playback, TTS, ASR.",
    no_args_is_help=True,
)

# 核心音频抽象槽位 + 未注册时的原因说明.
_SLOTS: list[tuple[str, Type[INSTANCE], str]] = [
    ("tts", TTS, ""),
    ("speech", Speech, ""),
    ("player", StreamAudioPlayer, ""),
    ("capture", AudioCaptureSource, "in mode HOST layer only — not in project container"),
    ("asr", ASR, "no provider registered"),
]


@audio_app.command("contracts")
def contracts() -> None:
    """List the IoC provider backing each core audio abstraction (no instantiation)."""
    matrix = Matrix.new("audio_cli", category="cli")
    con = matrix.container

    rows = []
    for slot, abstract, note in _SLOTS:
        try:
            provider = con.get_provider(abstract)
        except Exception as e:
            rows.append([slot, abstract.__name__, "—", f"get_provider error: {type(e).__name__} {e}"])
            continue
        if provider is None:
            rows.append([slot, abstract.__name__, "—", note or "no provider registered"])
        else:
            rows.append([slot, abstract.__name__, type(provider).__name__, "OK"])

    print_simple_table(
        data=rows,
        headers=["Slot", "Contract", "Provider", "Status"],
        title="audio contracts",
    )
    echo("")
    print_info(
        "Result is scoped to the active mode (`moss --mode <name>`). "
        "Use `moss manifests providers` for the full provider view."
    )


# --- play — 标准 tune / wav 播放, 验证体感, 支持中断与波形 ---

_WAVE_BARS = "▁▂▃▄▅▆▇█"


def _sparkline(samples, max_chars: int = 80) -> str:
    """把 PlaybackSample 的 rms_db 映射成文本波形条. 过长时按宽度采样."""
    if not samples:
        return "(no samples)"
    dbs = [s.rms_db for s in samples]
    if len(dbs) > max_chars:
        idx = np.linspace(0, len(dbs) - 1, max_chars).astype(int)
        dbs = [dbs[i] for i in idx]
    out = []
    for db in dbs:
        level = int((max(-60.0, min(0.0, db)) + 60.0) / 60.0 * (len(_WAVE_BARS) - 0.01))
        out.append(_WAVE_BARS[level])
    return "".join(out)


def _fragments(pcm: np.ndarray, rate: int, frag_ms: int = 100) -> list[np.ndarray]:
    """来源粒度切片 — CLI 作为来源按片段喂入, 每个片段一个 PlaybackSample.

    player 自己负责重采样与底层帧切分; 这里的切片只为波形观测,
    语义与 TTS 逐片段产出一致 (fragment_id 递增).
    """
    step = max(int(rate * frag_ms / 1000), 1)
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def _synthesize_tune(seconds: float, rate: int) -> np.ndarray:
    """温和的 C 大调和弦 pad — 基频 + 柔和泛音, 缓慢颤音/振幅起伏, 淡入淡出.

    目标是"不太难听"的体感测试音, 不是旋律. 峰值控制在 0.5, 听感不刺耳.
    """
    notes = [(261.63, 0.30), (329.63, 0.24), (392.00, 0.24), (523.25, 0.12)]
    n = int(seconds * rate)
    t = np.arange(n) / rate
    wave = np.zeros(n)
    for f, a in notes:
        vib = 0.6 * np.sin(2 * np.pi * 0.35 * t)  # 缓慢颤音, 声音"活"一点
        phase = 2 * np.pi * (f + vib) * t
        wave += a * np.sin(phase) + a * 0.25 * np.sin(2 * phase)
    wave *= 0.9 + 0.1 * np.sin(2 * np.pi * 1.1 * t)  # 慢振幅起伏
    attack = max(int(0.02 * rate), 1)
    release = max(int(0.12 * rate), 1)
    if n > attack + release:
        env = np.ones(n)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[-release:] *= np.linspace(1.0, 0.0, release)
        wave *= env
    peak = float(np.max(np.abs(wave))) or 1.0
    wave = wave / peak * 0.5
    return (wave * 32767).astype(np.int16)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """读 WAV PCM — 用标准库, 不引入具体播放实现. 多声道下混到单声道."""
    with _wave.open(str(path), "rb") as w:
        params = w.getparams()
        nch, sampwidth, rate, nframes = params[:4]
        raw = w.readframes(nframes)
    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) * 256
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        data = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return data, rate


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
    from rich.panel import Panel
    from rich.text import Text

    t = Text()
    t.append(f"stream={s.stream_id or '-'}  fragment={s.fragment_id or '-'}\n", style="dim")
    t.append(f"duration={s.duration:.2f}s  rms={s.rms_db:.1f}dB  peak={s.peak:.3f}\n")
    bar_w = 28
    lvl = int((max(-60.0, min(0.0, s.rms_db)) + 60.0) / 60.0 * bar_w)
    t.append("rms  " + "█" * lvl + "░" * (bar_w - lvl) + "\n", style="yellow")
    t.append(f"pcm={pcm_sz}B @{s.sample_rate}Hz", style="dim")
    console.print(Panel(t, title="playback sample", border_style="cyan"))


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


async def _spectrogram_play(player, pcm: np.ndarray, rate: int, stream_id: str, observed: list) -> None:
    """human 模式: 按片段喂入, 逐帧收集 PlaybackSample, 播放后渲染频谱谱面."""
    for i, frag in enumerate(_fragments(pcm, rate)):
        player.add(frag, audio_type=AudioFormat.PCM_S16LE, rate=rate, stream_id=stream_id, fragment_id=str(i))
    await player.wait_play_done()


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
    from rich.panel import Panel
    from rich.text import Text

    t = Text()
    t.append(f"fragments: {len(observed)}  ", style="dim")
    t.append(f"duration: {total:.2f}s\n")
    t.append(spectro, style="yellow")
    console.print(Panel(t, title="playback spectrogram", border_style="cyan"))


@audio_app.command("play")
def play(
    seconds: float = typer.Option(3.0, "--seconds", "-s", help="Standard tune duration in seconds (ignored when --file is given)."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Play a WAV file instead of the standard tune."),
) -> None:
    """Play a standard tune (or a WAV file) to test audible playback feel. Ctrl+C to interrupt.

    Human 模式: 播放中实时波形面板 (进度 + rms 波形). --ai 模式: 单帧样本摘要.
    """
    matrix = Matrix.new("audio_play", category="cli")
    matrix.run(lambda m: _async_play(m, seconds=seconds, file=file))


async def _async_play(matrix, *, seconds: float, file: Optional[Path]) -> None:
    player = matrix.container.get(StreamAudioPlayer)
    if player is None:
        print_error("StreamAudioPlayer not registered — run `moss audio contracts` to see what's available.")
        return

    if file is not None:
        if not file.exists():
            print_error(f"file not found: {file}")
            return
        pcm, rate = _read_wav(file)
        source = str(file)
    else:
        rate = 44100
        pcm = _synthesize_tune(seconds, rate)
        source = "tune"
    if len(pcm) == 0:
        print_warning("no audio to play")
        return
    total = len(pcm) / rate

    observed: list = []
    interrupted = False
    await player.start()
    unsub = player.observe(observed.append)
    try:
        stream_id = f"cli-{int(time.time())}"
        if is_ai_mode():
            # --ai 等价 -w 关闭: 整段单次 add, 单帧样本摘要.
            player.add(
                pcm,
                audio_type=AudioFormat.PCM_S16LE,
                rate=rate,
                stream_id=stream_id,
                fragment_id="0",
            )
            await player.wait_play_done()
        else:
            await _spectrogram_play(player, pcm, rate, stream_id, observed)
    except asyncio.CancelledError:
        # 用户 Ctrl+C — 经 matrix.run 生命周期以 CancelledError 送达, 立即掐断播放.
        interrupted = True
        await player.clear()
    finally:
        unsub()
        await player.close()

    if interrupted:
        print_warning("interrupted — playback stopped")
    else:
        print_success(f"playback finished: {source} {total:.2f}s")
    if is_ai_mode():
        _report_playback_sample(observed, total)
    else:
        _report_spectrogram(observed, total)
